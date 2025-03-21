#!/usr/bin/env python
"""
2__BulkPriceDownloader.py - OPTIMIZED VERSION
=============================================

INTERACTIVE BROKERS (IBKR) STOCK DATA DOWNLOADER - PERFORMANCE-FOCUSED
----------------------------------------------------------------------
This optimized version prioritizes download speed with aggressive connection reuse,
parallel processing, and minimal overhead. It's designed to download daily OHLCV data
rapidly for large sets of tickers.

KEY OPTIMIZATIONS:
-----------------
1. Aggressive connection reuse with smart pooling
2. Streamlined data processing (TRADES data only)
3. Parallel processing with higher worker counts
4. Reduced API overhead and rate limiting
5. Smart error handling with automatic retries
6. Memory optimizations for large ticker sets
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
import random
import warnings
import sys
from collections import deque
from functools import lru_cache
from Util import get_logger # Suppress warnings


warnings.filterwarnings('ignore')

# Import ib_insync with asyncio event loop fix
from ib_insync import IB, Contract, util, ibcontroller

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Determine the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the working directory to the script's directory
os.chdir(script_dir)

# Configuration
BASE_DIRECTORY = script_dir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'PriceData')
RF_PREDICTIONS_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'RFpredictions') 
LOG_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'logging')
TICKERS_CIK_DIRECTORY = os.path.join(BASE_DIRECTORY, 'Data', 'TickerCikData')
LOG_FILE = os.path.join(LOG_DIRECTORY, '2__BulkPriceDownloader.log')
PROGRESS_FILE = os.path.join(LOG_DIRECTORY, 'download_progress.json')

# OPTIMIZED SETTINGS: Streamlined for performance
# Only download TRADES data (normal OHLCV)
DATA_TYPES = ['TRADES']

# Define timeframes - focusing on daily data only
DAILY_BAR_SIZE = '1 day'
DAILY_DURATION = '3 Y'

# OPTIMIZED: Performance parameters for faster downloads
DEFAULT_BATCH_SIZE = 48       # Process more tickers per batch
MAX_WORKERS_PER_BATCH = 16    # Higher concurrent downloads
REQUEST_DELAY = 0.1           # Minimal delay between requests
BATCH_PAUSE = 0.5             # Shorter pause between batches
CONNECTION_POOL_SIZE = 4     # Maximum connections to maintain
REQUEST_TIMEOUT = 10          # Timeout for requests in seconds

# Ensure necessary directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
os.makedirs(RF_PREDICTIONS_DIRECTORY, exist_ok=True) 

# Set up logging - minimize logging overhead for speed

class ConnectionPool:
    """
    Advanced connection pool with smart resource management and load balancing.
    Uses a connection queue for efficient reuse and parallel processing.
    """
    def __init__(self, host='127.0.0.1', port=7497, max_connections=CONNECTION_POOL_SIZE):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.connection_queue = deque()
        self.active_connections = {}
        self.connection_count = 0
        self.lock = asyncio.Lock()
        self.next_client_id = random.randint(1000, 8000)
        
        # Connection metrics
        self.requests_count = 0
        self.connections_created = 0
        self.connections_failed = 0
        self.requests_per_connection = {}
        
        # Print initialized message
        print(f"Connection pool initialized with max {max_connections} connections")
        print(f"Host: {host}, Port: {port}")
        print("=" * 80)
        
    async def get_connection(self):
        """Get a connection from the pool or create a new one if needed."""
        async with self.lock:
            # Try to get a valid connection from the queue
            while self.connection_queue:
                tag, ib = self.connection_queue.popleft()
                
                # Check if the connection is still valid
                if ib is not None and ib.isConnected():
                    # tqdm.write(f"Reusing connection {tag}")
                    return tag, ib
                else:
                    # Connection was dropped, clean it up
                    try:
                        if ib is not None:
                            await ib.disconnectAsync()
                    except:
                        pass
                    
                    # Remove from active connections if it's there
                    if tag in self.active_connections:
                        del self.active_connections[tag]
                    if tag in self.requests_per_connection:
                        del self.requests_per_connection[tag]
            
            # Need to create a new connection
            return await self._create_new_connection()
    
    async def _create_new_connection(self):
        """Create a new IBKR connection."""
        client_id = self.next_client_id
        self.next_client_id = (self.next_client_id + 1) % 9000 + 1000
        tag = f"conn_{self.connection_count}"
        self.connection_count += 1
        
        # Create a new connection
        ib = IB()
        try:
            # Print status message
            logger.info(f"Creating new connection {tag} (client ID: {client_id})")
            
            # Connect with timeout
            await ib.connectAsync(self.host, self.port, clientId=client_id, timeout=REQUEST_TIMEOUT)
            
            # Verify the connection
            if not ib.isConnected():
                raise Exception("Connection failed to establish")
                
            # Register the new connection
            self.connections_created += 1
            self.active_connections[tag] = ib
            self.requests_per_connection[tag] = 0
            
            logger.info(f"Successfully created connection {tag}")
            return tag, ib
            
        except Exception as e:
            self.connections_failed += 1
            error_msg = str(e)
            logger.error(f"Failed to create connection {tag}: {error_msg}")
            tqdm.write(f"✗ Failed to create connection {tag}: {error_msg}")
            
            try:
                if ib:
                    await ib.disconnectAsync()
            except:
                pass
                
            # If we hit the maximum client ID limit, wait longer
            if "already in use" in error_msg.lower():
                await asyncio.sleep(1.0)  # Longer pause for ID conflicts
            else:
                await asyncio.sleep(0.5)  # Brief pause for other errors
                
            # Make sure we're not stuck in an infinite loop
            if self.connections_failed > 100:
                tqdm.write("⚠️ Too many connection failures. Check your TWS/Gateway settings.")
                raise Exception("Too many connection failures")
                
            # Try again with a different client ID
            return await self._create_new_connection()
    
    async def release_connection(self, tag, ib, force_disconnect=False):
        """Return a connection to the pool or close it if needed."""
        if ib is None:
            return
            
        async with self.lock:
            # Safety check - make sure tag and ib are valid
            if not tag or not isinstance(tag, str):
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
                
            # Check if still connected
            if not ib.isConnected() or force_disconnect:
                # Connection is not useful anymore, remove it
                if tag in self.active_connections:
                    del self.active_connections[tag]
                if tag in self.requests_per_connection:
                    del self.requests_per_connection[tag]
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
            
            # If we have too many connections, close this one
            if len(self.connection_queue) >= self.max_connections:
                if tag in self.active_connections:
                    del self.active_connections[tag]
                if tag in self.requests_per_connection:
                    del self.requests_per_connection[tag]
                try:
                    await ib.disconnectAsync()
                except:
                    pass
                return
            
            # Return connection to the pool
            self.connection_queue.append((tag, ib))
    
    async def record_request(self, tag):
        """Record metrics about a request to inform load balancing."""
        async with self.lock:
            self.requests_count += 1
            if tag in self.requests_per_connection:
                self.requests_per_connection[tag] += 1
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self.lock:
            # Close all queued connections
            while self.connection_queue:
                tag, ib = self.connection_queue.popleft()
                try:
                    await ib.disconnectAsync()
                except:
                    pass
            
            # Close all active connections
            for tag, ib in list(self.active_connections.items()):
                try:
                    await ib.disconnectAsync()
                except:
                    pass
            
            self.active_connections = {}
            self.requests_per_connection = {}
            
    def get_stats(self):
        """Get statistics about the connection pool."""
        return {
            "total_requests": self.requests_count,
            "connections_created": self.connections_created,
            "connections_failed": self.connections_failed,
            "active_connections": len(self.active_connections),
            "queued_connections": len(self.connection_queue),
            "requests_per_connection": self.requests_per_connection
        }

# Global connection pool
connection_pool = None

@lru_cache(maxsize=1000)
def get_contract(ticker, security_type='STK', exchange='SMART', currency='USD'):
    """Create and cache an IBKR contract object for the specified ticker."""
    return Contract(symbol=ticker, secType=security_type, exchange=exchange, currency=currency)

async def download_historical_data(connection_info, ticker, duration_str, use_rth=True):
    """Download historical data for a ticker with minimal overhead."""
    tag, ib = connection_info
    fail_reason = None
    
    try:
        # Create contract
        contract = get_contract(ticker)
        
        # Record this request
        await connection_pool.record_request(tag)
        
        # Request contract details to verify the contract
        contract_details = await ib.reqContractDetailsAsync(contract)
        if not contract_details:
            return None, "No contract details found"
        
        # Request historical data (async)
        data_type = 'TRADES'  # Only fetch TRADES data
        bars = await ib.reqHistoricalDataAsync(
            contract=contract,
            endDateTime='',  # empty for latest data
            durationStr=duration_str,
            barSizeSetting=DAILY_BAR_SIZE,
            whatToShow=data_type,
            useRTH=use_rth,
            formatDate=1  # yyyyMMdd format
        )
        
        if not bars:
            return None, "No historical data returned"
        
        # Convert to DataFrame
        df = util.df(bars)
        
        if df.empty:
            return None, "Empty DataFrame returned"
        
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
        
        # Add ticker information
        df['Ticker'] = ticker
        
        return df, None
        
    except Exception as e:
        error_message = str(e)
        
        # Check for typical rate limit errors
        if "pacing violation" in error_message.lower():
            fail_reason = "PACING"
        elif "connection reset" in error_message.lower() or "timeout" in error_message.lower():
            fail_reason = "CONNECTION"
        elif "session is connected from a different IP address" in error_message:
            fail_reason = "SESSION"
        else:
            fail_reason = "OTHER"
            
        logger.error(f"Error downloading data for {ticker}: {error_message}")
        return None, fail_reason

def save_to_parquet(df, file_path):
    """Save DataFrame to parquet file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save DataFrame to parquet with compression
        df.to_parquet(file_path, index=False, compression='snappy')
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        return False

def format_yahoo_style_data(df):
    """Format data to match Yahoo Finance format with minimal processing."""
    try:
        if df is None or df.empty:
            return None
            
        # Create a copy to avoid modifying the original
        yahoo_df = df.copy()
        
        # Keep only the columns we need (matching Yahoo Finance format)
        yahoo_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        if all(col in yahoo_df.columns for col in yahoo_columns):
            yahoo_df = yahoo_df[yahoo_columns]
        else:
            missing_cols = [col for col in yahoo_columns if col not in yahoo_df.columns]
            logger.warning(f"Missing columns in data: {missing_cols}")
            return None
        
        # Sort by date
        yahoo_df.sort_values('Date', inplace=True)
        
        # Round values to reduce file size
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        yahoo_df[numeric_cols] = yahoo_df[numeric_cols].round(4)
        
        return yahoo_df
        
    except Exception as e:
        logger.error(f"Error formatting data to Yahoo style: {str(e)}")
        return None

def save_progress(processed_tickers, total_tickers, remaining_tickers=None, pbar=None):
    """Save download progress to JSON file without interrupting the progress bar."""
    try:
        progress = {
            'processed_tickers': processed_tickers,
            'total_tickers': total_tickers,
            'remaining_tickers': remaining_tickers if remaining_tickers else [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Use atomic writing to prevent corruption
        temp_file = PROGRESS_FILE + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(progress, f)
        
        # Atomic replacement
        if os.path.exists(PROGRESS_FILE):
            os.replace(temp_file, PROGRESS_FILE)
        else:
            os.rename(temp_file, PROGRESS_FILE)
    except Exception as e:
        logger.error(f"Error saving progress: {str(e)}")

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

def find_latest_ticker_cik_file(directory):
    """Find the latest TickerCIKs file based on modification time."""
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Latest TickerCIKs file found: {latest_file}")
    return latest_file

def get_existing_tickers(data_dir):
    """Extract ticker symbols from existing data files in the specified directory."""
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []
    
    try:
        for file in os.listdir(data_dir):
            match = pattern.match(file)
            if match and match.group(1):  # Ensure we have a non-empty ticker
                ticker = match.group(1)
                # Verify it's a valid ticker (not empty or malformed)
                if ticker and isinstance(ticker, str) and len(ticker) > 0:
                    tickers.append(ticker)
    except Exception as e:
        print(f"Error getting existing tickers: {str(e)}")
        
    # Remove any potential None values
    tickers = [t for t in tickers if t]
    
    print(f"Found {len(tickers)} existing tickers in {data_dir} for RefreshMode.")
    return tickers


async def process_ticker(ticker, use_rth=True, daily_duration=DAILY_DURATION, max_retries=3):
    """Process a single ticker with automatic retries."""
    global connection_pool
    
    # Safety checks
    if not ticker or not isinstance(ticker, str):
        logger.error(f"Invalid ticker: {ticker}")
        return False
        
    retry_count = 0
    retry_delay = 1  # Initial retry delay in seconds
    success = False
    
    while retry_count <= max_retries and not success:
        # Get a connection from the pool
        try:
            connection_info = await connection_pool.get_connection()
            tag, ib = connection_info
            
            if not ib or not ib.isConnected():
                logger.warning(f"Got invalid connection for {ticker}, retrying...")
                retry_count += 1
                await asyncio.sleep(0.5)
                continue
                
            # Download the data
            df, fail_reason = await download_historical_data(
                connection_info=connection_info,
                ticker=ticker,
                duration_str=daily_duration,
                use_rth=use_rth
            )
            
            # If successful, save the data
            if df is not None:
                # Format to Yahoo style
                yahoo_df = format_yahoo_style_data(df)
                if yahoo_df is not None:
                    # Save the data
                    file_path = os.path.join(DATA_DIRECTORY, f"{ticker}.parquet")
                    if save_to_parquet(yahoo_df, file_path):
                        success = True
            
            # Handle different failure reasons
            if not success:
                if fail_reason == "PACING":
                    # Pacing violation, retry after a longer delay
                    retry_delay = min(retry_delay * 2, 5)  # Exponential backoff, max 5 seconds
                    # Force disconnect this connection due to pacing violation
                    await connection_pool.release_connection(tag, ib, force_disconnect=True)
                elif fail_reason == "SESSION" or fail_reason == "CONNECTION":
                    # Session or connection error, retry with a new connection
                    await connection_pool.release_connection(tag, ib, force_disconnect=True)
                    retry_delay = 0.5  # Short delay for connection issues
                else:
                    # For other failures, standard retry
                    await connection_pool.release_connection(tag, ib)
            else:
                # Success case - return connection to pool
                await connection_pool.release_connection(tag, ib)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            # Connection might not exist if we failed before getting one
            if 'tag' in locals() and 'ib' in locals():
                try:
                    await connection_pool.release_connection(tag, ib, force_disconnect=True)
                except:
                    pass
        
        # If not successful, retry
        if not success:
            retry_count += 1
            if retry_count <= max_retries:
                await asyncio.sleep(retry_delay)  # Wait before retry
    
    return success

async def process_batch(tickers, use_rth=True, daily_duration=DAILY_DURATION, 
                       pbar=None, semaphore=None):
    """Process a batch of tickers with controlled concurrency."""
    success_count = 0
    fail_count = 0
    
    # Create tasks for concurrent processing
    async def process_with_semaphore(ticker):
        try:
            async with semaphore:
                success = await process_ticker(
                    ticker=ticker,
                    use_rth=use_rth,
                    daily_duration=daily_duration
                )
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    postfix_str = str(pbar.postfix) if hasattr(pbar, 'postfix') and pbar.postfix else ""
                    current_fails = 0
                    if postfix_str and "failed=" in postfix_str:
                        try:
                            current_fails = int(postfix_str.split('failed=')[1].split(',')[0])
                        except (IndexError, ValueError):
                            current_fails = 0
                    
                    if not success:
                        current_fails += 1
                    success_rate = (pbar.n - current_fails) / pbar.n * 100 if pbar.n > 0 else 0
                    pbar.set_postfix(failed=current_fails, success_rate=f"{success_rate:.1f}%", last=ticker)
                
                return success
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            if pbar:
                pbar.update(1)
            return False
    
    # Create a semaphore to limit concurrency
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_WORKERS_PER_BATCH)
    
    # Process tickers concurrently with better error handling
    tasks = []
    for ticker in tickers:
        if ticker:  # Only process non-None tickers
            tasks.append(process_with_semaphore(ticker))
    
    if not tasks:
        return 0, 0  # No valid tickers to process
        
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
    
    return success_count, fail_count

async def process_all_tickers(tickers, host='127.0.0.1', port=7497, batch_size=DEFAULT_BATCH_SIZE, 
                             daily_duration=DAILY_DURATION, use_rth=True, resume=False,
                             max_workers=MAX_WORKERS_PER_BATCH):
    """Process all tickers in batches with connection pooling."""
    global connection_pool
    
    # Initialize connection pool
    connection_pool = ConnectionPool(host=host, port=port, max_connections=CONNECTION_POOL_SIZE)
    
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
    
    # Clear terminal for clean display
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Show header with summary information
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - OPTIMIZED VERSION")
    print("="*80)
    print(f"Tickers: {len(remaining_tickers)} pending, {len(processed_tickers)} completed")
    print(f"Batch size: {batch_size}, Workers: {max_workers}")
    print(f"Connection pool size: {CONNECTION_POOL_SIZE}")
    print("="*80)
    
    # Create a progress bar for ALL tickers
    pbar = tqdm(total=len(remaining_tickers), desc="Download Progress", 
                unit="ticker", ncols=100, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)
    
    # Process tickers in batches
    for i in range(0, len(remaining_tickers), batch_size):
        batch = remaining_tickers[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size
        
        # Update progress bar description
        pbar.set_description(f"Batch {batch_num}/{total_batches}")
        
        # Process the batch
        batch_success, batch_fail = await process_batch(
            tickers=batch,
            use_rth=use_rth,
            daily_duration=daily_duration,
            pbar=pbar,
            semaphore=semaphore
        )
        
        # Track progress
        success_count += batch_success
        fail_count += batch_fail
        
        # Update processed tickers list
        batch_processed = [t for idx, t in enumerate(batch) if idx < batch_success]
        processed_tickers.extend(batch_processed)
        remaining_after_batch = remaining_tickers[i+batch_size:]
        
        # Save progress after each batch
        save_progress(processed_tickers, total_count, remaining_after_batch)
        
        # Print connection pool stats periodically
        if batch_num % 2 == 0 or batch_num == total_batches:
            stats = connection_pool.get_stats()
            logger.info(f"Connection pool stats: {stats['active_connections']} active, {stats['queued_connections']} queued, {stats['total_requests']} requests")
        
        # Pause between batches
        if batch_num < total_batches:
            await asyncio.sleep(BATCH_PAUSE)
    
    # Close the progress bar
    pbar.close()
    
    # Close all connections
    await connection_pool.close_all()
    
    # Final summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Total processed: {success_count + fail_count} tickers")
    print(f"Successful: {success_count} ({success_count/(success_count+fail_count)*100:.1f}%)")
    print(f"Failed: {fail_count}")
    
    # Print final connection pool stats
    stats = connection_pool.get_stats()
    print("\nConnection pool stats:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Connections created: {stats['connections_created']}")
    print(f"Connections failed: {stats['connections_failed']}")
    print("="*80)
    
    return success_count, fail_count, total_count

def display_header(args):
    """Display a clean header with configuration information."""
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display header
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER - OPTIMIZED VERSION")
    print("="*80)
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Configuration info
    config_lines = [
        f"Host: {args.host}, Port: {args.port}",
        f"Batch size: {args.batch_size}, Workers: {args.max_workers}",
        f"Connection pool size: {args.pool_size}",
        f"Delays: Request={args.request_delay}s, Batch pause={args.batch_pause}s",
        f"Data: Daily={args.daily_duration} (Regular Trading Hours: {'Yes' if args.use_rth else 'No'})",
        f"Mode: {'Single ticker' if args.ticker else 'ColdStart' if args.ColdStart else 'Refresh' if args.RefreshMode else 'Unknown'}"
    ]
    
    for line in config_lines:
        print(line)
    
    print("="*80)
    
    # Mode-specific messages
    if args.ticker:
        print(f"Processing single ticker: {args.ticker}")
    elif args.RefreshMode:
        print("Running in REFRESH MODE: Updating existing ticker data")
    elif args.ColdStart:
        print(f"Running in {'RESUME' if args.Resume else 'COLDSTART'} MODE: Processing all tickers")
        if args.ClearOldData:
            print("⚠️ Clearing old data before starting")
    
    print("="*80)
    return

def main(logger):
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
    parser.add_argument('--daily-duration', type=str, default=DAILY_DURATION, help=f'Daily data duration (default: {DAILY_DURATION})')
    parser.add_argument('--use-rth', action='store_true', help='Use regular trading hours only')
    # Performance settings
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Number of tickers to process in each batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS_PER_BATCH, help=f'Maximum number of concurrent workers per batch (default: {MAX_WORKERS_PER_BATCH})')
    parser.add_argument('--pool-size', type=int, default=CONNECTION_POOL_SIZE, help=f'Size of the connection pool (default: {CONNECTION_POOL_SIZE})')
    parser.add_argument('--request-delay', type=float, default=REQUEST_DELAY, help=f'Delay between requests in seconds (default: {REQUEST_DELAY})')
    parser.add_argument('--batch-pause', type=float, default=BATCH_PAUSE, help=f'Pause between batches in seconds (default: {BATCH_PAUSE})')
    args = parser.parse_args()
    
    # Update global settings based on arguments
    globals()['REQUEST_DELAY'] = args.request_delay
    globals()['BATCH_PAUSE'] = args.batch_pause
    globals()['CONNECTION_POOL_SIZE'] = args.pool_size
    
    # Display header with configuration information
    display_header(args)
    
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
        # Refresh existing tickers from RF_PREDICTIONS_DIRECTORY instead of DATA_DIRECTORY
        print("Running in Refresh Mode: Refreshing data for tickers from RF_PREDICTIONS_DIRECTORY.")
        tickers = get_existing_tickers(RF_PREDICTIONS_DIRECTORY)
        if not tickers:
            print("No tickers found in RF_PREDICTIONS_DIRECTORY for RefreshMode. Exiting.")
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
     
    try:
        # Process tickers
        start_time = time.time()
        
        if len(tickers) == 1:
            # Single ticker mode - simpler handling
            print(f"Processing single ticker {tickers[0]}...")
            success = asyncio.run(process_ticker(
                ticker=tickers[0],
                use_rth=args.use_rth,
                daily_duration=args.daily_duration
            ))
            
            success_count = 1 if success else 0
            fail_count = 0 if success else 1
            total_count = 1
        else:
            # Batch processing mode with connection pooling and advanced concurrency
            success_count, fail_count, total_count = asyncio.run(process_all_tickers(
                tickers=tickers,
                host=args.host,
                port=args.port,
                batch_size=args.batch_size,
                daily_duration=args.daily_duration,
                use_rth=args.use_rth,
                resume=args.Resume,
                max_workers=args.max_workers
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
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Cleaning up...")
        if 'connection_pool' in globals() and connection_pool is not None:
            asyncio.run(connection_pool.close_all())
        print("Cleanup complete. Exiting.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Ensure connection pool is closed
        if 'connection_pool' in globals() and connection_pool is not None:
            try:
                asyncio.run(connection_pool.close_all())
                print("All connections closed.")
            except:
                print("Error while closing connections.")

if __name__ == "__main__":
    logger = get_logger(script_name="1__TickerDownloader")
    main(logger)