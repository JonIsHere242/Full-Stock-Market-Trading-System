#!/root/root/miniconda4/envs/tf/bin/python
import datetime
import os
import requests
import pandas as pd
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys

# Import the enhanced logging functions
from Util import get_logger, LogPerformance

CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs_{date}.parquet",
    "user_agent": "MarketAnalysis Masamunex9000@gmail.com"
}

def setup_args():
    parser = argparse.ArgumentParser(description="Download and convert Ticker CIK data.")
    parser.add_argument("--ImmediateDownload", action='store_true', 
                        help="Download the file immediately without waiting for the scheduled time.")
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="Increase output verbosity")
    args = parser.parse_args()
    return args

def download_and_convert_ticker_cik_file(logger):
    try:
        logger.debug("Starting download_and_convert_ticker_cik_file function")
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        parquet_file_path = CONFIG["parquet_file_path"].format(date=current_date)
        logger.debug(f"Parquet file path: {parquet_file_path}")

        # Use the performance context manager
        with LogPerformance("SEC ticker data download", logger=logger):
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            headers = {
                'User-Agent': CONFIG["user_agent"],
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            logger.debug(f"Making request to {CONFIG['url']}")
            response = session.get(CONFIG["url"], headers=headers, timeout=30)
            response.raise_for_status()
            logger.debug("Request successful")

            json_data = response.json()
            logger.debug(f"Received JSON data with {len(json_data['data'])} entries")
            df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
            logger.debug(f"Created DataFrame with shape {df.shape}")

            df = df[df['exchange'].notna()]
            df = df[~df['exchange'].isin(['OTC', 'CBOE'])]
            logger.debug(f"Filtered DataFrame, new shape: {df.shape}")

            os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)
            df.to_parquet(parquet_file_path, index=False)
            logger.info(f"File saved successfully: {parquet_file_path}")
            
            # Return processed data count for statistics
            return df.shape[0]

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    args = setup_args()
    
    # Set up the logger with the new system
    logger = get_logger(debug=args.verbose)
    
    logger.info("Script started")
    if args.ImmediateDownload:
        logger.info("Immediate download requested")
        try:
            ticker_count = download_and_convert_ticker_cik_file(logger)
            logger.info(f"Download completed successfully with {ticker_count} tickers processed")
        except Exception as e:
            logger.error(f"Download failed: {e}")
    else:
        logger.info("No immediate download requested")
    logger.info("Script completed")