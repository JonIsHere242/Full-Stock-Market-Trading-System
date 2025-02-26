#!/usr/bin/env python
import yfinance as yf
import pandas as pd
import os
import logging
import time
from datetime import datetime, timedelta
import argparse
import glob
import re
from tqdm import tqdm
import pytz
import sys
import traceback

"""
This script downloads stock data from Yahoo Finance based on the presence of ticker files in the FINAL_DATA_DIRECTORY.
It supports both initial downloads (ColdStart) and refreshing existing data by appending the latest missing data.
Only high-quality data that extends back to at least January 1, 2022, is processed and saved.
"""

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the script's directory
os.chdir(script_dir)

# Directory and File Configurations
FINAL_DATA_DIRECTORY = os.path.join(script_dir, "Data", "RFpredictions")
DATA_DIRECTORY = os.path.join(script_dir, 'Data', 'PriceData')
TICKERS_CIK_DIRECTORY = os.path.join(script_dir, 'Data', 'TickerCikData')
LOG_FILE = os.path.join(script_dir, "Data", "logging", "2__BulkPriceDownloader.log")
RATE_LIMIT = 1.0  # seconds between downloads
START_DATE = "2022-01-01"  # Updated start date to ensure data quality
MIN_EARLIEST_DATE = datetime(2022, 3, 1).date()
MIN_DAYS_OF_DATA = 400  # Minimum required days of data

# Ensure necessary directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(FINAL_DATA_DIRECTORY, exist_ok=True)  # Ensure RFpredictions directory exists

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.debug(f"Data directory: {os.path.abspath(DATA_DIRECTORY)}")
    logging.debug(f"Ticker CIK directory: {os.path.abspath(TICKERS_CIK_DIRECTORY)}")
    logging.debug(f"Final Data directory: {os.path.abspath(FINAL_DATA_DIRECTORY)}")

def get_existing_tickers_final(data_dir):
    """
    Extract ticker symbols from existing data files in the FINAL_DATA_DIRECTORY.
    This assumes that each file is named as {ticker}.parquet.
    """
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            tickers.append(match.group(1))
    logging.info(f"Found {len(tickers)} existing tickers in FINAL_DATA_DIRECTORY for RefreshMode.")
    return tickers

def find_latest_ticker_cik_file(directory):
    """Find the latest TickerCIKs file based on modification time."""
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    logging.info(f"Latest TickerCIKs file found: {latest_file}")
    return latest_file

def clear_old_data(data_dir):
    """Remove all .csv and .parquet files from the data directory."""
    removed_files = 0
    for file in os.listdir(data_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(data_dir, file))
            removed_files += 1
    logging.info(f"Cleared {removed_files} old data files from {data_dir}.")

def get_last_trading_day(reference_datetime=None):
    eastern = pytz.timezone('US/Eastern')
    if reference_datetime is None:
        reference_datetime = datetime.now(eastern)
    else:
        if reference_datetime.tzinfo is None:
            reference_datetime = eastern.localize(reference_datetime)
        else:
            reference_datetime = reference_datetime.astimezone(eastern)

    weekday = reference_datetime.weekday()  # Monday=0, Sunday=6
    current_time = reference_datetime.time()
    data_available_time = datetime.strptime("22:00", "%H:%M").time()  # Data available after 10 PM Eastern

    logging.debug(f"Reference datetime (US/Eastern): {reference_datetime}")
    logging.debug(f"Weekday: {weekday}, Current Time: {current_time}")
    logging.debug(f"Data Availability Cutoff Time: {data_available_time}")

    if weekday < 5:
        if current_time >= data_available_time:
            # After 10 PM on a weekday, data for today should be available
            last_trading_day = reference_datetime.date()
        else:
            # Before 10 PM on a weekday, data for today may not be available
            last_trading_day = reference_datetime.date() - timedelta(days=1)
            # If yesterday was a weekend, adjust to last weekday
            while last_trading_day.weekday() >= 5:
                last_trading_day -= timedelta(days=1)
    else:
        # Today is weekend, find the last weekday
        last_trading_day = reference_datetime.date()
        while last_trading_day.weekday() >= 5:
            last_trading_day -= timedelta(days=1)
    logging.debug(f"Determined last trading day as: {last_trading_day}")
    return last_trading_day

def fetch_and_save_stock_data(tickers, data_dir, start_date=None, end_date=None, rate_limit=RATE_LIMIT, refresh=False):
    download_count = 0
    wait_time = rate_limit
    timer_start = time.time()

    logging.info(f"Fetching data from {start_date} to {end_date}")

    for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
        file_path = os.path.join(data_dir, f"{ticker}.parquet")
        
        try:
            if refresh and os.path.exists(file_path):
                # Read existing data to find the latest date
                existing_df = pd.read_parquet(file_path)
                if 'Date' in existing_df.columns:
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
                    latest_date = existing_df['Date'].max().date()
                elif isinstance(existing_df.index, pd.DatetimeIndex):
                    existing_df = existing_df.reset_index()
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
                    latest_date = existing_df['Date'].max().date()
                else:
                    logging.warning(f"File {ticker}.parquet does not have a 'Date' column or DateTime index. Skipping refresh.")
                    continue

                # Ensure the existing data extends back to at least MIN_EARLIEST_DATE
                earliest_date_existing = existing_df['Date'].min().date()

                if earliest_date_existing > MIN_EARLIEST_DATE:
                    logging.warning(f"{ticker} existing data starts at {earliest_date_existing}, which is after {MIN_EARLIEST_DATE}. Skipping refresh.")
                    continue

                # Determine the start date for new data
                new_start_date = latest_date + timedelta(days=1)
                if new_start_date > end_date.date():
                    # No new data to fetch
                    logging.info(f"{ticker} is already up-to-date.")
                    continue
            else:
                # Initial download
                existing_df = None
                new_start_date = start_date

            # Fetch data from Yahoo Finance using the adjusted end date
            stock_data = yf.download(
                ticker, 
                start=new_start_date,
                end=end_date.strftime('%Y-%m-%d'), 
                interval="1d", 
                progress=False
            )

            if stock_data.empty:
                logging.warning(f"No new data found for {ticker}.")
                continue

            # Reset index to ensure 'Date' is a column
            stock_data.reset_index(inplace=True)

            # Ensure 'Date' column exists and is datetime
            if 'Date' not in stock_data.columns:
                logging.error(f"Downloaded data for {ticker} does not contain 'Date' column. Skipping.")
                continue
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')

            # Drop rows with NaNs in 'Date' or critical columns
            stock_data.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

            if refresh and existing_df is not None:
                # Append new data to existing data
                combined_data = pd.concat([existing_df, stock_data], ignore_index=True)
                combined_data.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                combined_data.sort_values('Date', inplace=True)
            else:
                combined_data = stock_data.copy()

            # Perform data quality checks on the combined data
            # Ensure data extends back to at least MIN_EARLIEST_DATE
            earliest_date = combined_data['Date'].min().date()
            if earliest_date > MIN_EARLIEST_DATE:
                logging.warning(f"{ticker} combined data starts at {earliest_date}, which is after {MIN_EARLIEST_DATE}. Skipping.")
                continue

            # Ensure sufficient data points (e.g., at least MIN_DAYS_OF_DATA days)
            days_of_data = (combined_data['Date'].max().date() - combined_data['Date'].min().date()).days
            if days_of_data < MIN_DAYS_OF_DATA:
                logging.warning(f"{ticker} has less than {MIN_DAYS_OF_DATA} days of data ({days_of_data} days). Skipping.")
                continue

            # Additional Data Integrity Checks
            # 1. No NaNs in critical columns
            if combined_data[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any().any():
                logging.warning(f"{ticker} has NaNs in critical columns after combining data. Skipping.")
                continue

            # 2. Check if the data is continuous (no missing trading days)
            combined_data_sorted = combined_data.sort_values('Date')
            all_dates = pd.date_range(start=combined_data_sorted['Date'].min(), end=combined_data_sorted['Date'].max(), freq='B')  # 'B' frequency for business days
            if not combined_data_sorted['Date'].isin(all_dates).all():
                missing_days = all_dates.difference(combined_data_sorted['Date'])
                logging.warning(f"{ticker} is missing {len(missing_days)} business days. Skipping.")
                continue

            # Round data to 5 decimal places
            combined_data = combined_data.round(5)

            # Save the combined data
            combined_data.to_parquet(file_path, index=False)

            download_count += 1

            if download_count % 100 == 0:
                elapsed_time = time.time() - timer_start
                logging.info(f"Downloaded data for {download_count} tickers, taking {elapsed_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"Error downloading data for {ticker}: {e}")
            logging.debug(traceback.format_exc())
            continue

        finally:
            # Enforce rate limit between downloads
            logging.debug(f"Sleeping for {wait_time} seconds to respect rate limit.")
            time.sleep(wait_time)

    logging.info(f"Total tickers processed: {download_count}")

if __name__ == "__main__":
    setup_logging()
    
    logging.info(f"Python Version: {sys.version}")
    logging.info(f"Timezone: {time.tzname}")
    
    parser = argparse.ArgumentParser(description="Download or refresh stock data from Yahoo Finance.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")
    parser.add_argument("--RefreshMode", action='store_true', help="Refresh existing data by appending the latest missing data.")
    parser.add_argument("--ColdStart", action='store_true', help="Initial download of all tickers from the CIK file.")
    
    args = parser.parse_args()
    
    if args.ClearOldData:
        clear_old_data(DATA_DIRECTORY)
    
    if args.RefreshMode and args.ColdStart:
        logging.error("Cannot use both --RefreshMode and --ColdStart simultaneously. Exiting.")
        exit(1)
    
    if args.RefreshMode:
        logging.info("Running in Refresh Mode: Refreshing data for tickers from FINAL_DATA_DIRECTORY.")
        tickers = get_existing_tickers_final(FINAL_DATA_DIRECTORY)
        if not tickers:
            logging.error("No tickers found in FINAL_DATA_DIRECTORY for RefreshMode. Exiting.")
            exit(1)
    elif args.ColdStart:
        logging.info("Running in ColdStart Mode: Downloading data for all tickers from the CIK file.")
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            logging.error("No TickerCIKs file found. Exiting.")
            exit(1)
        tickers_df = pd.read_parquet(ticker_cik_file)
        if 'ticker' not in tickers_df.columns:
            logging.error("The TickerCIKs file does not contain a 'ticker' column. Exiting.")
            exit(1)
        tickers = tickers_df['ticker'].dropna().unique().tolist()
    else:
        logging.error("No mode selected. Please use either --RefreshMode or --ColdStart.")
        parser.print_help()
        exit(1)
    
    expected_latest_date = get_last_trading_day()
    end_date = expected_latest_date + timedelta(days=1)

    # Download or refresh the stock data
    fetch_and_save_stock_data(
        tickers,
        DATA_DIRECTORY,
        start_date=START_DATE,
        end_date=end_date,
        rate_limit=RATE_LIMIT,
        refresh=args.RefreshMode
    )
    
    logging.info("Data download and refresh process completed.")
