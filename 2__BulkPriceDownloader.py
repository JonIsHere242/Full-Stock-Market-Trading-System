#!/usr/bin/env python
"""
2__BulkPriceDownloader.py
=========================

INTERACTIVE BROKERS (IBKR) STOCK DATA DOWNLOADER - OPTIMIZED VERSION
--------------------------------------------------------------------
This script downloads historical stock data from Interactive Brokers (IBKR) with faster parallel processing
and resume capability. It offers both single-ticker and batch processing modes with built-in rate limiting
to prevent IBKR pacing violations.

PREREQUISITES:
-------------
1. IBKR Trader Workstation (TWS) or IB Gateway must be running
2. API connections must be enabled in TWS/Gateway settings
3. ib_insync package must be installed: pip install ib_insync
4. Required directory structure (Data/PriceData, Data/TickerCikData, Data/logging)

COMMAND LINE USAGE:
------------------

1. DOWNLOAD A SINGLE TICKER:
   python 2__BulkPriceDownloader.py --ticker AAPL

2. COLD START MODE (Download all tickers from most recent TickerCIK file):
   python 2__BulkPriceDownloader.py --ColdStart

3. REFRESH MODE (Update existing tickers in RFpredictions directory):
   python 2__BulkPriceDownloader.py --RefreshMode

4. RESUME MODE (Continue downloading from where you left off):
   python 2__BulkPriceDownloader.py --ColdStart --Resume

5. CLEAR OLD DATA:
   python 2__BulkPriceDownloader.py --ColdStart --ClearOldData

6. CUSTOMIZE BATCH SIZE AND WORKERS:
   python 2__BulkPriceDownloader.py --ColdStart --batch-size 30 --max-workers 10

7. OPTIMIZE SPEED WITH RATE LIMITS:
   python 2__BulkPriceDownloader.py --ColdStart --request-delay 0.5 --batch-pause 3

8. SKIP MINUTE DATA (Speeds up processing significantly):
   python 2__BulkPriceDownloader.py --ColdStart --skip-minute-data

9. CONNECT TO REAL TRADING ACCOUNT (Default is paper trading):
   python 2__BulkPriceDownloader.py --ColdStart --port 7496

10. USE REGULAR TRADING HOURS ONLY (Excludes pre/post market):
    python 2__BulkPriceDownloader.py --ColdStart --use-rth

11. ADJUST DATA DURATIONS:
    python 2__BulkPriceDownloader.py --ColdStart --daily-duration "2 Y" --minute-duration "3 D"

12. FULL EXAMPLES:

    # Fast download with resume capability:
    python 2__BulkPriceDownloader.py --ColdStart --Resume --batch-size 30 --max-workers 10 --request-delay 0.5 --batch-pause 3 --skip-minute-data

    # Complete refresh of all existing tickers with minute data:
    python 2__BulkPriceDownloader.py --RefreshMode --batch-size 20 --daily-duration "3 Y" --minute-duration "5 D"

    # Download a single stock with full detail:
    python 2__BulkPriceDownloader.py --ticker MSFT --daily-duration "5 Y" --minute-duration "2 W"

ALL AVAILABLE ARGUMENTS:
----------------------
  --host HOST           TWS/Gateway host (default: 127.0.0.1)
  --port PORT           TWS/Gateway port (7497 for paper, 7496 for real)
  --ticker TICKER       Single ticker symbol to process
  --RefreshMode         Refresh existing data by appending the latest missing data
  --ColdStart           Initial download of all tickers from the CIK file
  --Resume              Resume downloading from last saved progress
  --ClearOldData        Clears any existing data in the output directory
  --daily-duration DAILY_DURATION
                        Daily data duration (default: 3 Y)
  --minute-duration MINUTE_DURATION
                        Minute data duration (default: 1 W)
  --skip-minute-data    Skip downloading minute data
  --use-rth             Use regular trading hours only
  --batch-size BATCH_SIZE
                        Number of tickers to process in each batch (default: 50)
  --max-workers MAX_WORKERS
                        Maximum number of concurrent workers per batch (default: 10)
  --request-delay REQUEST_DELAY
                        Delay between requests in seconds (default: 0.5)
  --batch-pause BATCH_PAUSE
                        Pause between batches in seconds (default: 5)

OUTPUT FILES:
------------
1. Yahoo Finance-compatible files: Data/PriceData/{ticker}.parquet
2. Enhanced daily data with bid/ask: Data/PriceData/{ticker}/{ticker}_DAILY_ENHANCED.parquet
3. Minute data: Data/PriceData/{ticker}/{ticker}_MINUTE_ENHANCED.parquet
4. Log file: Data/logging/IBKR_Downloader.log
5. Progress file: Data/logging/download_progress.json

PARALLEL PROCESSING:
------------------
This script implements parallel ticker processing to speed up downloads:
- Multiple tickers processed concurrently within each batch
- Optimized connection management
- Progress tracking for resume capability
"""

import os
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
from tqdm import tqdm
import traceback
import glob
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import ib_insync with asyncio event loop fix
from ib_insync import IB, Contract, util

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the script's directory
os.chdir(script_dir)

# Configuration
BASE_DIRECTORY = script_dir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'PriceData')
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'logging')
TICKERS_CIK_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'TickerCikData')
LOG_FILE = os.path.join(LOG_DIRECTORY, '2__BulkPriceDownloader.log')
PROGRESS_FILE = os.path.join(LOG_DIRECTORY, 'download_progress.json')

# Define data types to download
DATA_TYPES = [
    'TRADES',      # Regular OHLCV data (primary)
    'BID',         # Bid prices
    'ASK',         # Ask prices
    'MIDPOINT'     # Midpoint between bid and ask
]

# Define timeframes
DAILY_BAR_SIZE = '1 day'
MINUTE_BAR_SIZE = '1 min'

# Define durations
DAILY_DURATION = '3 Y'       # 3 years for daily data
MINUTE_DURATION = '1 W'      # 1 week for minute data (to keep it manageable)

# Define rate limiting parameters
DEFAULT_BATCH_SIZE = 32      # Default batch size (increased from 10)
MAX_WORKERS_PER_BATCH = 32   # Number of concurrent workers per batch
REQUEST_DELAY = 0.5          # Reduced delay between requests in seconds (from 1.5)
BATCH_PAUSE = 5              # Reduced pause between batches in seconds (from 15)

# Define data quality parameters
START_DATE = "2022-01-01"
MIN_EARLIEST_DATE = datetime(2022, 3, 1).date()
MIN_DAYS_OF_DATA = 400  # Minimum required days of data

# Ensure necessary directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.ERROR,  # Changed from INFO to ERROR
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        # Removed StreamHandler to disable terminal output
    ]
)
logger = logging.getLogger(__name__)

# Create a console logger for tqdm compatibility
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only show errors in console
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def save_progress(processed_tickers, total_tickers, remaining_tickers=None):
    """Save download progress to JSON file."""
    progress = {
        'processed_tickers': processed_tickers,
        'total_tickers': total_tickers,
        'remaining_tickers': remaining_tickers if remaining_tickers else [],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)
    print(f"Progress saved: {len(processed_tickers)}/{total_tickers} tickers processed")



def load_progress():
    """Load download progress from JSON file."""
    if not os.path.exists(PROGRESS_FILE):
        return [], []
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        print(f"Loaded progress: {len(progress['processed_tickers'])}/{progress.get('total_tickers', 0)} tickers processed")
        return progress['processed_tickers'], progress.get('remaining_tickers', [])
    except Exception as e:
        print(f"Could not load progress file: {str(e)}")
        logger.error(f"Could not load progress file: {str(e)}")
        return [], []


def connect_to_ibkr(host='127.0.0.1', port=7497, client_id=None, timeout=20):
    """Connect to Interactive Brokers TWS or Gateway."""
    if client_id is None:
        client_id = int(time.time()) % 9000 + 1000  # Generate a random-ish client ID
    
    ib = IB()
    
    try:
        print(f"Connecting to IBKR at {host}:{port} with client ID {client_id}")
        ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
        
        if ib.isConnected():
            print("Successfully connected to IBKR")
            return ib
        else:
            logger.error("Failed to connect to IBKR")
            print("Failed to connect to IBKR")
            return None
    except Exception as e:
        logger.error(f"Error connecting to IBKR: {str(e)}")
        print(f"Error connecting to IBKR: {str(e)}")
        return None

def get_contract(ticker, security_type='STK', exchange='SMART', currency='USD'):
    """Create an IBKR contract object for the specified ticker."""
    return Contract(symbol=ticker, secType=security_type, exchange=exchange, currency=currency)

async def download_historical_data(ib, ticker, data_type, bar_size, duration_str, use_rth=True):
    """Download historical data for a ticker with specified type and bar size."""
    try:
        # Create contract
        contract = get_contract(ticker)
        
        # Request contract details to verify the contract
        contract_details = await ib.reqContractDetailsAsync(contract)
        if not contract_details:
            logger.warning(f"No contract details found for {ticker}")
            return None
        
        logger.info(f"Downloading {duration_str} of {bar_size} {data_type} data for {ticker}")
        
        # Request historical data (async)
        bars = await ib.reqHistoricalDataAsync(
            contract=contract,
            endDateTime='',  # empty for latest data
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=data_type,
            useRTH=use_rth,
            formatDate=1  # yyyyMMdd format
        )
        
        if not bars:
            logger.warning(f"No historical data returned for {ticker} ({data_type}, {bar_size})")
            return None
        
        # Convert to DataFrame
        df = util.df(bars)
        
        if df.empty:
            logger.warning(f"Empty DataFrame returned for {ticker} ({data_type}, {bar_size})")
            return None
        
        # Adjust column names to match Yahoo Finance format
        df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'average': 'Average',
            'barCount': 'BarCount'
        }, inplace=True)
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add data type to the DataFrame for reference
        df['DataType'] = data_type
        
        logger.info(f"Downloaded {len(df)} bars for {ticker} ({data_type}, {bar_size})")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {data_type} data for {ticker} ({bar_size}): {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def save_to_parquet(df, file_path):
    """Save DataFrame to parquet file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save DataFrame to parquet
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def format_yahoo_style_data(df_trades):
    """
    Format TRADES data to match the Yahoo Finance format we were using before.
    This ensures compatibility with existing pipeline.
    """
    try:
        if df_trades is None or df_trades.empty:
            return None
            
        # Create a copy to avoid modifying the original
        yahoo_df = df_trades.copy()
        
        # Keep only the columns we need (matching Yahoo Finance format)
        yahoo_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in yahoo_df.columns for col in yahoo_columns):
            yahoo_df = yahoo_df[yahoo_columns]
        else:
            missing_cols = [col for col in yahoo_columns if col not in yahoo_df.columns]
            logger.warning(f"Missing columns in data: {missing_cols}")
            return None
        
        # Sort by date
        yahoo_df.sort_values('Date', inplace=True)
        
        # Check data quality
        if yahoo_df.empty:
            logger.warning("DataFrame is empty after formatting")
            return None
            
        # Round values to 5 decimal places (like Yahoo Finance)
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        yahoo_df[numeric_cols] = yahoo_df[numeric_cols].round(5)
        
        return yahoo_df
        
    except Exception as e:
        logger.error(f"Error formatting data to Yahoo style: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def merge_all_data_types(dataframes_dict, ticker, timeframe):
    """
    Merge multiple data type dataframes into a wide format DataFrame.
    
    Args:
        dataframes_dict: Dictionary mapping data types to dataframes
        ticker: Ticker symbol
        timeframe: 'DAILY' or 'MINUTE'
    
    Returns:
        Wide-format DataFrame with all data types as columns
    """
    try:
        if not dataframes_dict:
            logger.warning(f"No dataframes to merge for {ticker} {timeframe}")
            return None
        
        # Start with the TRADES dataframe if available
        base_data_type = 'TRADES' if 'TRADES' in dataframes_dict else list(dataframes_dict.keys())[0]
        base_df = dataframes_dict[base_data_type].copy()
        
        # Keep only these columns from the base dataframe
        base_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Average' in base_df.columns:
            base_columns.append('Average')
        if 'BarCount' in base_df.columns:
            base_columns.append('BarCount')
        
        # Filter to only include columns we want
        base_df = base_df[base_columns]
        
        # Process other data types
        for data_type, df in dataframes_dict.items():
            if data_type == base_data_type:
                continue
            
            # Create prefix for column names
            prefix = f"{data_type}_"
            
            # Create a copy of the dataframe with renamed columns
            df_copy = df.copy()
            rename_dict = {}
            for col in df_copy.columns:
                if col != 'Date' and col not in ['DataType', 'BarSize']:
                    rename_dict[col] = f"{prefix}{col}"
            
            df_copy = df_copy.rename(columns=rename_dict)
            
            # Merge with base dataframe
            base_df = pd.merge(base_df, df_copy[['Date'] + list(rename_dict.values())], on='Date', how='outer')
        
        # Sort by date
        base_df.sort_values('Date', inplace=True)
        
        # Add ticker and timeframe columns
        base_df['Ticker'] = ticker
        base_df['Timeframe'] = timeframe
        
        return base_df
    
    except Exception as e:
        logger.error(f"Error merging dataframes for {ticker} {timeframe}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def find_latest_ticker_cik_file(directory):
    """Find the latest TickerCIKs file based on modification time."""
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Latest TickerCIKs file found: {latest_file}")
    return latest_file


def get_existing_tickers(data_dir):
    """
    Extract ticker symbols from existing data files in the DATA_DIRECTORY.
    This assumes that each file is named as {ticker}.parquet.
    """
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            tickers.append(match.group(1))
    print(f"Found {len(tickers)} existing tickers in DATA_DIRECTORY for RefreshMode.")
    return tickers

async def process_ticker(ticker, daily_duration=DAILY_DURATION, minute_duration=MINUTE_DURATION, 
                         include_minute_data=False, use_rth=True, host='127.0.0.1', port=7497, 
                         delay=REQUEST_DELAY, connection=None):
    """
    Process a single ticker to download and save data:
    1. Download daily data for all data types
    2. Save Yahoo-style daily data (for compatibility with existing pipeline)
    3. Save enhanced daily data with all data types
    4. Optionally download and save minute data (recent only)
    """
    # Create result variables
    daily_dataframes = {}
    minute_dataframes = {}
    success = False
    need_to_disconnect = False
    
    # Connect to IBKR if a connection wasn't provided
    ib = connection
    if not ib:
        ib = connect_to_ibkr(host=host, port=port)
        need_to_disconnect = True
        
    if not ib:
        logger.error(f"Failed to connect to IBKR. Cannot process {ticker}.")
        return False
    
    try:
        logger.info(f"Processing ticker: {ticker}")
        
        # Process daily data
        logger.info(f"Downloading daily data for {ticker}")
        
        # Prepare tasks for all data types
        tasks = []
        for data_type in DATA_TYPES:
            task = download_historical_data(
                ib=ib,
                ticker=ticker,
                data_type=data_type,
                bar_size=DAILY_BAR_SIZE,
                duration_str=daily_duration,
                use_rth=use_rth
            )
            tasks.append((data_type, task))
        
        # Process all data types concurrently
        for data_type, task in tasks:
            try:
                df = await task
                if df is not None and not df.empty:
                    daily_dataframes[data_type] = df
                await asyncio.sleep(delay)  # Small delay between tasks
            except Exception as e:
                logger.error(f"Error processing {ticker} daily {data_type}: {str(e)}")
        
        # Save Yahoo-style data (TRADES only, for compatibility)
        if 'TRADES' in daily_dataframes:
            yahoo_style_df = format_yahoo_style_data(daily_dataframes['TRADES'])
            if yahoo_style_df is not None:
                # Check data quality
                if not yahoo_style_df.empty:
                    earliest_date = yahoo_style_df['Date'].min().date()
                    if earliest_date > MIN_EARLIEST_DATE:
                        logger.warning(f"{ticker} data starts at {earliest_date}, which is after {MIN_EARLIEST_DATE}. Skipping.")
                    else:
                        # Save the data only to PriceData directory
                        yahoo_file_path = os.path.join(DATA_DIRECTORY, f"{ticker}.parquet")
                        save_to_parquet(yahoo_style_df, yahoo_file_path)
                        
                        # We have successfully saved Yahoo-style data
                        success = True
        else:
            logger.warning(f"No TRADES data available for {ticker}. Cannot create Yahoo-style data.")
        
        # Save enhanced daily data (all data types merged)
        if daily_dataframes:
            daily_enhanced_df = merge_all_data_types(daily_dataframes, ticker, 'DAILY')
            if daily_enhanced_df is not None:
                enhanced_daily_path = os.path.join(DATA_DIRECTORY, ticker, f"{ticker}_DAILY_ENHANCED.parquet")
                save_to_parquet(daily_enhanced_df, enhanced_daily_path)
        
        # Process minute data (if requested)
        if include_minute_data and success:  # Only get minute data if daily data was successful
            logger.info(f"Downloading minute data for {ticker} (last {minute_duration})")
            
            # Prepare tasks for all data types
            minute_tasks = []
            for data_type in DATA_TYPES:
                task = download_historical_data(
                    ib=ib,
                    ticker=ticker,
                    data_type=data_type,
                    bar_size=MINUTE_BAR_SIZE,
                    duration_str=minute_duration,
                    use_rth=use_rth
                )
                minute_tasks.append((data_type, task))
            
            # Process all data types concurrently
            for data_type, task in minute_tasks:
                try:
                    df = await task
                    if df is not None and not df.empty:
                        minute_dataframes[data_type] = df
                    await asyncio.sleep(delay)  # Small delay between tasks
                except Exception as e:
                    logger.error(f"Error processing {ticker} minute {data_type}: {str(e)}")
            
            # Save minute data (all data types merged)
            if minute_dataframes:
                minute_enhanced_df = merge_all_data_types(minute_dataframes, ticker, 'MINUTE')
                if minute_enhanced_df is not None:
                    enhanced_minute_path = os.path.join(DATA_DIRECTORY, ticker, f"{ticker}_MINUTE_ENHANCED.parquet")
                    save_to_parquet(minute_enhanced_df, enhanced_minute_path)
        
    except Exception as e:
        logger.error(f"Error processing ticker {ticker}: {str(e)}")
        logger.debug(traceback.format_exc())
        success = False
    
    finally:
        # Disconnect only if we created the connection
        if need_to_disconnect and ib and ib.isConnected():
            ib.disconnect()
            logger.info(f"Disconnected from IBKR after processing {ticker}")
        
        return success

async def process_ticker_batch(tickers, connection, host='127.0.0.1', port=7497, 
                             daily_duration=DAILY_DURATION, minute_duration=MINUTE_DURATION,
                             include_minute_data=False, use_rth=True):
    """Process multiple tickers concurrently with a single connection."""
    success_count = 0
    fail_count = 0
    
    try:
        # Process each ticker in parallel using asyncio.gather
        tasks = []
        for ticker in tickers:
            task = process_ticker(
                ticker=ticker,
                daily_duration=daily_duration,
                minute_duration=minute_duration,
                include_minute_data=include_minute_data,
                use_rth=use_rth,
                connection=connection
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        for result in results:
            if isinstance(result, Exception):
                fail_count += 1
                logger.error(f"Task failed with exception: {result}")
            elif result is True:
                success_count += 1
            else:
                fail_count += 1
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        logger.debug(traceback.format_exc())
    
    return success_count, fail_count


async def process_batch(tickers, host='127.0.0.1', port=7497, 
                      daily_duration=DAILY_DURATION, minute_duration=MINUTE_DURATION,
                      include_minute_data=False, use_rth=True, max_workers=MAX_WORKERS_PER_BATCH,
                      quiet_mode=False):
    """Process a batch of tickers with a single connection using parallel workers."""
    if not tickers:
        return 0, 0
    
    # Create IBKR connection for this batch
    print(f"Connecting to IBKR for batch of {len(tickers)} tickers...")
    ib = connect_to_ibkr(host=host, port=port)
    if not ib:
        logger.error("Failed to connect to IBKR for batch processing.")
        print("Failed to connect to IBKR for batch processing.")
        return 0, len(tickers)
    
    success_count = 0
    fail_count = 0
    
    try:
        # Split the batch into smaller chunks for thread workers
        chunks = [tickers[i:i + max_workers] for i in range(0, len(tickers), max_workers)]
        
        # Create a progress bar for this batch (only if not in quiet mode)
        if not quiet_mode:
            chunk_pbar = tqdm(chunks, desc="Chunk Progress", unit="chunk", 
                            ncols=100, position=1, leave=False)
        
        for i, chunk in enumerate(chunks):
            if not quiet_mode:
                chunk_pbar.set_description(f"Processing {len(chunk)} tickers")

            chunk_success, chunk_fail = await process_ticker_batch(
                tickers=chunk,
                connection=ib,
                host=host,
                port=port,
                daily_duration=daily_duration,
                minute_duration=minute_duration,
                include_minute_data=include_minute_data,
                use_rth=use_rth
            )
            
            success_count += chunk_success
            fail_count += chunk_fail
            
            # Update progress bar postfix (only if not in quiet mode)
            if not quiet_mode:
                chunk_pbar.set_postfix(
                    success=chunk_success, 
                    fail=chunk_fail,
                    total_success=success_count,
                    total_fail=fail_count
                )
            
            # Small pause between chunks
            await asyncio.sleep(1)
        
        # Close the progress bar (only if not in quiet mode)
        if not quiet_mode:
            chunk_pbar.close()
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        print(f"Error in batch processing: {str(e)}")
        logger.debug(traceback.format_exc())
        fail_count += len(tickers) - success_count
    
    finally:
        # Disconnect from IBKR
        if ib and ib.isConnected():
            ib.disconnect()
            print("Disconnected from IBKR after batch processing")
        
        return success_count, fail_count



async def process_all_tickers(tickers, host='127.0.0.1', port=7497, batch_size=DEFAULT_BATCH_SIZE, 
                             daily_duration=DAILY_DURATION, minute_duration=MINUTE_DURATION,
                             include_minute_data=False, use_rth=True, resume=False,
                             max_workers=MAX_WORKERS_PER_BATCH, quiet_mode=False):
    """Process all tickers in batches to respect rate limits with resume support."""
    processed_tickers = []
    remaining_tickers = tickers.copy()
    
    # If resume mode, load previously processed tickers
    if resume:
        processed_tickers, loaded_remaining = load_progress()
        if loaded_remaining:
            remaining_tickers = loaded_remaining
        else:
            # Filter out already processed tickers
            remaining_tickers = [t for t in tickers if t not in processed_tickers]
        
        print(f"Resuming download: {len(processed_tickers)} already processed, {len(remaining_tickers)} remaining")
    
    success_count = len(processed_tickers)
    fail_count = 0
    total_count = len(tickers)
    
    print(f"Processing {len(remaining_tickers)} tickers in batches of {batch_size}")
    
    # Create progress bar for overall progress
    pbar = tqdm(total=len(remaining_tickers), desc="Overall Progress", 
                unit="ticker", ncols=100, position=0, leave=True)
    pbar.update(0)  # Initialize the bar
    
    # Process tickers in batches
    for i in range(0, len(remaining_tickers), batch_size):
        batch = remaining_tickers[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size
        
        # Update progress bar description with batch info
        pbar.set_description(f"Batch {batch_num}/{total_batches}")
        
        # Process the batch
        batch_success, batch_fail = await process_batch(
            tickers=batch,
            host=host,
            port=port,
            daily_duration=daily_duration,
            minute_duration=minute_duration,
            include_minute_data=include_minute_data,
            use_rth=use_rth,
            max_workers=max_workers,
            quiet_mode=quiet_mode
        )
        
        success_count += batch_success
        fail_count += batch_fail
        
        # Update processed and remaining tickers
        newly_processed = batch[:batch_success]  # Approximate - assuming first batch_success tickers succeeded
        processed_tickers.extend(newly_processed)
        remaining_after_batch = remaining_tickers[i+batch_size:]
        
        # Save progress after each batch
        save_progress(processed_tickers, total_count, remaining_after_batch)
        
        # Update progress bar with number of successful downloads in this batch
        pbar.update(len(newly_processed))
        
        # Set description to show success rate
        success_rate = (success_count / (success_count + fail_count)) * 100 if (success_count + fail_count) > 0 else 0
        pbar.set_postfix(success_rate=f"{success_rate:.1f}%", successful=success_count, failed=fail_count)
        
        # Print batch completion info
        print(f"Batch {batch_num}/{total_batches} completed: "
              f"{batch_success}/{len(batch)} successful, {batch_fail} failed")
        
        # Pause between batches to respect rate limits
        if batch_num < total_batches:
            print(f"Pausing for {BATCH_PAUSE} seconds before next batch")
            await asyncio.sleep(BATCH_PAUSE)
    
    # Close the progress bar
    pbar.close()
    
    return success_count, fail_count, total_count


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Download stock data from IBKR with optimized parallel processing')
    
    # IBKR connection settings
    parser.add_argument('--host', type=str, default='127.0.0.1', help='TWS/Gateway host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7497, help='TWS/Gateway port (7497 for paper, 7496 for real)')
    
    # Mode settings
    parser.add_argument('--ticker', type=str, help='Single ticker symbol to process')
    parser.add_argument('--RefreshMode', action='store_true', help='Refresh existing data by appending the latest missing data.')
    parser.add_argument('--ColdStart', action='store_true', help='Initial download of all tickers from the CIK file.')
    parser.add_argument('--ClearOldData', action='store_true', help='Clears any existing data in the output directory.')
    parser.add_argument('--Resume', action='store_true', help='Resume from the last saved progress point.')
    
    # Data settings
    parser.add_argument('--daily-duration', type=str, default=DAILY_DURATION, 
                        help=f'Daily data duration (default: {DAILY_DURATION})')
    parser.add_argument('--minute-duration', type=str, default=MINUTE_DURATION, 
                        help=f'Minute data duration (default: {MINUTE_DURATION})')
    parser.add_argument('--skip-minute-data', action='store_true', 
                        help='Skip downloading minute data')
    parser.add_argument('--use-rth', action='store_true', 
                        help='Use regular trading hours only')
    
    # Performance settings
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f'Number of tickers to process in each batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS_PER_BATCH,
                        help=f'Maximum number of concurrent workers per batch (default: {MAX_WORKERS_PER_BATCH})')
    parser.add_argument('--request-delay', type=float, default=REQUEST_DELAY,
                        help=f'Delay between requests in seconds (default: {REQUEST_DELAY})')
    parser.add_argument('--batch-pause', type=int, default=BATCH_PAUSE,
                        help=f'Pause between batches in seconds (default: {BATCH_PAUSE})')
    
    # Display settings
    parser.add_argument('--quiet', action='store_true', 
                        help='Disable the mid-batch progress bar, showing only the overall progress')
    
    args = parser.parse_args()
    
    # Update global variables via globals() dictionary
    globals()['REQUEST_DELAY'] = args.request_delay
    globals()['BATCH_PAUSE'] = args.batch_pause
    
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER")
    print("="*80)
    print(f"Python Version: {os.sys.version}")
    print(f"Batch size: {args.batch_size}, Max workers: {args.max_workers}")
    print(f"Request delay: {REQUEST_DELAY}s, Batch pause: {BATCH_PAUSE}s")
    print(f"Minute data: {'Skipped' if args.skip_minute_data else 'Included'}")
    print(f"Display mode: {'Quiet (overall progress only)' if args.quiet else 'Full (all progress bars)'}")
    print("="*80)
    
    # Clear old data if requested
    if args.ClearOldData:
        print("Clearing old data...")
        for file in os.listdir(DATA_DIRECTORY):
            file_path = os.path.join(DATA_DIRECTORY, file)
            if os.path.isfile(file_path) and (file.endswith('.csv') or file.endswith('.parquet')):
                os.remove(file_path)
        print("Old data cleared")
    
    # Get ticker list
    tickers = []
    
    if args.ticker:
        # Single ticker mode
        tickers = [args.ticker]
        print(f"Processing single ticker: {args.ticker}")
    elif args.RefreshMode and args.ColdStart:
        print("Cannot use both --RefreshMode and --ColdStart simultaneously. Exiting.")
        exit(1)
    elif args.RefreshMode:
        # Refresh existing tickers
        print("Running in Refresh Mode: Refreshing data for tickers from DATA_DIRECTORY.")
        tickers = get_existing_tickers(DATA_DIRECTORY)
        if not tickers:
            print("No tickers found in DATA_DIRECTORY for RefreshMode. Exiting.")
            exit(1)
    elif args.ColdStart or args.Resume:
        # Download all tickers from CIK file
        print(f"Running in {'Resume' if args.Resume else 'ColdStart'} Mode: Downloading data for all tickers from the CIK file.")
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            print("No TickerCIKs file found. Exiting.")
            exit(1)
        tickers_df = pd.read_parquet(ticker_cik_file)
        if 'ticker' not in tickers_df.columns:
            print("The TickerCIKs file does not contain a 'ticker' column. Exiting.")
            exit(1)
        tickers = tickers_df['ticker'].dropna().unique().tolist()
    else:
        print("No mode selected. Please use --ticker, --RefreshMode, --ColdStart, or --Resume.")
        parser.print_help()
        exit(1)
    
    print(f"Found {len(tickers)} tickers to process")
    
    # Process tickers
    start_time = time.time()
    
    if len(tickers) == 1:
        # Single ticker mode
        print(f"Processing single ticker {tickers[0]}...")
        success = asyncio.run(process_ticker(
            ticker=tickers[0],
            daily_duration=args.daily_duration,
            minute_duration=args.minute_duration,
            include_minute_data=not args.skip_minute_data,
            use_rth=args.use_rth,
            host=args.host,
            port=args.port,
            delay=REQUEST_DELAY
        ))
        
        success_count = 1 if success else 0
        fail_count = 0 if success else 1
        total_count = 1
    else:
        # Batch processing mode with resume support
        success_count, fail_count, total_count = asyncio.run(process_all_tickers(
            tickers=tickers,
            host=args.host,
            port=args.port,
            batch_size=args.batch_size,
            daily_duration=args.daily_duration,
            minute_duration=args.minute_duration,
            include_minute_data=not args.skip_minute_data,
            use_rth=args.use_rth,
            resume=args.Resume,
            max_workers=args.max_workers,
            quiet_mode=args.quiet
        ))
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Process completed in {elapsed_time/60:.2f} minutes")
    print(f"Total tickers: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed/Skipped: {fail_count}")
    print(f"Success rate: {(success_count/total_count)*100:.2f}%")
    
    if success_count > 0:
        print(f"Data saved to {DATA_DIRECTORY}")
    else:
        print("No tickers were successfully processed")
    print("="*80)

if __name__ == "__main__":
    main()