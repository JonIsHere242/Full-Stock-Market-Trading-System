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
DEFAULT_BATCH_SIZE = 10     # Default batch size (increased from 10)
MAX_WORKERS_PER_BATCH = 4   # Number of concurrent workers per batch
REQUEST_DELAY = 1.0         # Reduced delay between requests in seconds (from 1.5)
BATCH_PAUSE = 2.0           # Reduced pause between batches in seconds (from 15)

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



def save_progress(processed_tickers, total_tickers, remaining_tickers=None, pbar=None):
    """Save download progress to JSON file without interrupting the progress bar."""
    progress = {
        'processed_tickers': processed_tickers,
        'total_tickers': total_tickers,
        'remaining_tickers': remaining_tickers if remaining_tickers else [],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)
    
    # Use tqdm.write instead of print to avoid breaking progress bar
    if pbar:
        logger.info(f"Progress saved: {len(processed_tickers)}/{total_tickers} tickers processed")
    else:
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




class ConnectionMonitor:
    """
    Monitors connection metrics and provides detailed statistics about connection usage.
    """
    def __init__(self):
        self.connection_history = []
        self.active_connections = {}
        self.start_time = datetime.now()
        
    def register_connection(self, tag, client_id):
        """Register a new connection."""
        self.active_connections[tag] = {
            'client_id': client_id,
            'created_at': datetime.now(),
            'request_count': 0,
            'last_request_time': None,
            'max_request_interval': 0,
            'min_request_interval': float('inf'),
            'avg_request_interval': 0,
            'total_request_interval': 0,
            'request_intervals': [],
        }
        logger.info(f"Registered connection: {tag} (client_id: {client_id})")
        
    def log_request(self, tag):
        """Log a request made on a connection."""
        if tag not in self.active_connections:
            return
            
        conn = self.active_connections[tag]
        current_time = datetime.now()
        
        # Update request count
        conn['request_count'] += 1
        
        # Calculate request interval if not the first request
        if conn['last_request_time'] is not None:
            interval = (current_time - conn['last_request_time']).total_seconds()
            conn['total_request_interval'] += interval
            conn['request_intervals'].append(interval)
            
            # Update min/max intervals
            conn['max_request_interval'] = max(conn['max_request_interval'], interval)
            conn['min_request_interval'] = min(conn['min_request_interval'], interval)
            
            # Update average
            conn['avg_request_interval'] = conn['total_request_interval'] / len(conn['request_intervals'])
        
        # Update last request time
        conn['last_request_time'] = current_time
        
    def unregister_connection(self, tag):
        """Unregister a connection and move it to history."""
        if tag not in self.active_connections:
            return
            
        conn = self.active_connections[tag]
        
        # Calculate connection lifetime
        lifetime = (datetime.now() - conn['created_at']).total_seconds()
        
        # Create history record
        history_record = {
            'tag': tag,
            'client_id': conn['client_id'],
            'created_at': conn['created_at'],
            'closed_at': datetime.now(),
            'lifetime_seconds': lifetime,
            'request_count': conn['request_count'],
            'avg_request_interval': conn['avg_request_interval'],
            'min_request_interval': conn['min_request_interval'] if conn['min_request_interval'] != float('inf') else 0,
            'max_request_interval': conn['max_request_interval'],
            'requests_per_second': conn['request_count'] / lifetime if lifetime > 0 else 0
        }
        
        # Add to history and remove from active
        self.connection_history.append(history_record)
        del self.active_connections[tag]
        
        logger.info(f"Unregistered connection: {tag} (lifetime: {lifetime:.2f}s, requests: {history_record['request_count']})")
        
    def get_metrics(self):
        """Get comprehensive connection metrics."""
        current_time = datetime.now()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Count connections
        total_connections = len(self.connection_history) + len(self.active_connections)
        active_connections = len(self.active_connections)
        closed_connections = len(self.connection_history)
        
        # Calculate total requests
        active_requests = sum(conn['request_count'] for conn in self.active_connections.values())
        closed_requests = sum(record['request_count'] for record in self.connection_history)
        total_requests = active_requests + closed_requests
        
        # Calculate average lifetime
        if self.connection_history:
            avg_lifetime = sum(record['lifetime_seconds'] for record in self.connection_history) / len(self.connection_history)
            max_lifetime = max(record['lifetime_seconds'] for record in self.connection_history) if self.connection_history else 0
            min_lifetime = min(record['lifetime_seconds'] for record in self.connection_history) if self.connection_history else 0
        else:
            avg_lifetime = max_lifetime = min_lifetime = 0
            
        # Calculate average requests per connection
        avg_requests_per_conn = total_requests / total_connections if total_connections > 0 else 0
        
        # Calculate request rates
        requests_per_second = total_requests / uptime if uptime > 0 else 0
        
        # Get currently active connection stats
        active_conn_stats = []
        for tag, conn in self.active_connections.items():
            lifetime = (current_time - conn['created_at']).total_seconds()
            active_conn_stats.append({
                'tag': tag,
                'client_id': conn['client_id'],
                'lifetime_seconds': lifetime,
                'request_count': conn['request_count'],
                'requests_per_second': conn['request_count'] / lifetime if lifetime > 0 else 0
            })
        
        # Calculate request distribution
        request_counts = [record['request_count'] for record in self.connection_history]
        request_counts.extend([conn['request_count'] for conn in self.active_connections.values()])
        
        if request_counts:
            request_distribution = {
                'min': min(request_counts),
                'max': max(request_counts),
                'avg': sum(request_counts) / len(request_counts),
                'median': sorted(request_counts)[len(request_counts) // 2],
                'percentile_25': sorted(request_counts)[int(len(request_counts) * 0.25)],
                'percentile_75': sorted(request_counts)[int(len(request_counts) * 0.75)],
                'percentile_90': sorted(request_counts)[int(len(request_counts) * 0.90)] if len(request_counts) >= 10 else None,
                'percentile_95': sorted(request_counts)[int(len(request_counts) * 0.95)] if len(request_counts) >= 20 else None,
                'percentile_99': sorted(request_counts)[int(len(request_counts) * 0.99)] if len(request_counts) >= 100 else None,
            }
        else:
            request_distribution = {'min': 0, 'max': 0, 'avg': 0, 'median': 0, 
                                   'percentile_25': 0, 'percentile_75': 0, 
                                   'percentile_90': None, 'percentile_95': None, 'percentile_99': None}
        
        # Find busiest connection
        busiest_connection = None
        max_requests = 0
        for record in self.connection_history:
            if record['request_count'] > max_requests:
                max_requests = record['request_count']
                busiest_connection = record
                
        for tag, conn in self.active_connections.items():
            if conn['request_count'] > max_requests:
                max_requests = conn['request_count']
                lifetime = (current_time - conn['created_at']).total_seconds()
                busiest_connection = {
                    'tag': tag,
                    'client_id': conn['client_id'],
                    'created_at': conn['created_at'],
                    'lifetime_seconds': lifetime,
                    'request_count': conn['request_count'],
                    'requests_per_second': conn['request_count'] / lifetime if lifetime > 0 else 0,
                    'is_active': True
                }
        
        return {
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'uptime_seconds': uptime,
            'uptime_formatted': f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            'connection_counts': {
                'total': total_connections,
                'active': active_connections,
                'closed': closed_connections
            },
            'request_counts': {
                'total': total_requests,
                'from_active': active_requests,
                'from_closed': closed_requests
            },
            'connection_lifetime': {
                'avg_seconds': avg_lifetime,
                'min_seconds': min_lifetime,
                'max_seconds': max_lifetime,
                'avg_formatted': f"{int(avg_lifetime // 60)}m {int(avg_lifetime % 60)}s" if avg_lifetime else "0s"
            },
            'request_rates': {
                'requests_per_second': requests_per_second,
                'avg_requests_per_connection': avg_requests_per_conn
            },
            'request_distribution': request_distribution,
            'busiest_connection': busiest_connection,
            'active_connections': active_conn_stats
        }
    
    def log_metrics(self, level=logging.INFO):
        """Log comprehensive metrics."""
        metrics = self.get_metrics()
        
        # Format metrics for logging
        log_lines = [
            "="*80,
            "CONNECTION MONITOR METRICS",
            "="*80,
            f"Start time: {metrics['start_time']}",
            f"Uptime: {metrics['uptime_formatted']}",
            "",
            "CONNECTION STATISTICS:",
            f"  Total connections: {metrics['connection_counts']['total']}",
            f"  Active connections: {metrics['connection_counts']['active']}",
            f"  Closed connections: {metrics['connection_counts']['closed']}",
            "",
            "REQUEST STATISTICS:",
            f"  Total requests: {metrics['request_counts']['total']}",
            f"  Requests from active connections: {metrics['request_counts']['from_active']}",
            f"  Requests from closed connections: {metrics['request_counts']['from_closed']}",
            f"  Requests per second: {metrics['request_rates']['requests_per_second']:.2f}",
            f"  Avg requests per connection: {metrics['request_rates']['avg_requests_per_connection']:.2f}",
            "",
            "CONNECTION LIFETIME:",
            f"  Average lifetime: {metrics['connection_lifetime']['avg_formatted']}",
            f"  Min lifetime: {int(metrics['connection_lifetime']['min_seconds'])}s",
            f"  Max lifetime: {int(metrics['connection_lifetime']['max_seconds'])}s",
            "",
            "REQUEST DISTRIBUTION (requests per connection):",
            f"  Min: {metrics['request_distribution']['min']}",
            f"  25th percentile: {metrics['request_distribution']['percentile_25']}",
            f"  Median: {metrics['request_distribution']['median']}",
            f"  75th percentile: {metrics['request_distribution']['percentile_75']}",
            f"  Max: {metrics['request_distribution']['max']}",
        ]
        
        # Add 90th, 95th, 99th percentiles if available
        if metrics['request_distribution']['percentile_90'] is not None:
            log_lines.append(f"  90th percentile: {metrics['request_distribution']['percentile_90']}")
        if metrics['request_distribution']['percentile_95'] is not None:
            log_lines.append(f"  95th percentile: {metrics['request_distribution']['percentile_95']}")
        if metrics['request_distribution']['percentile_99'] is not None:
            log_lines.append(f"  99th percentile: {metrics['request_distribution']['percentile_99']}")
            
        # Add busiest connection info if available
        if metrics['busiest_connection']:
            bc = metrics['busiest_connection']
            log_lines.extend([
                "",
                "BUSIEST CONNECTION:",
                f"  Tag: {bc.get('tag', 'N/A')}",
                f"  Client ID: {bc.get('client_id', 'N/A')}",
                f"  Total requests: {bc.get('request_count', 0)}",
                f"  Lifetime: {int(bc.get('lifetime_seconds', 0))}s",
                f"  Requests per second: {bc.get('requests_per_second', 0):.2f}",
                f"  Status: {'Active' if bc.get('is_active', False) else 'Closed'}"
            ])
            
        # Add active connection details
        if metrics['active_connections']:
            log_lines.extend(["", "ACTIVE CONNECTIONS:"])
            for i, conn in enumerate(metrics['active_connections']):
                log_lines.extend([
                    f"  Connection #{i+1}:",
                    f"    Tag: {conn['tag']}",
                    f"    Client ID: {conn['client_id']}",
                    f"    Lifetime: {int(conn['lifetime_seconds'])}s",
                    f"    Request count: {conn['request_count']}",
                    f"    Requests per second: {conn['requests_per_second']:.2f}"
                ])
                
        log_lines.append("="*80)
        
        # Join all lines and log
        log_message = "\n".join(log_lines)
        logger.log(level, log_message)
        
        # Also print to console if needed
        if level <= logging.INFO:
            tqdm.write(log_message)
            
        return log_message

class IBKRConnectionManager:
    """
    Manages IBKR connections to ensure proper cleanup and prevent connection leaks.
    With enhanced monitoring metrics.
    """
    def __init__(self):
        self.active_connections = {}
        self.connection_count = 0
        self.setup_signal_handlers()
        self.monitor = ConnectionMonitor()
        logger.info("Connection manager initialized with monitoring")
        
    def setup_signal_handlers(self):
        """Set up signal handlers to ensure connections are closed on termination."""
        import signal
        import sys
        
        def signal_handler(sig, frame):
            tqdm.write("\n⚠️ Received termination signal. Cleaning up connections...")
            # Log metrics before disconnecting
            self.monitor.log_metrics()
            self.disconnect_all()
            tqdm.write("✓ All connections closed. Exiting.")
            sys.exit(0)
            
        # Register signal handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler) # Termination request
        
    def connect(self, host='127.0.0.1', port=7497, client_id=None, timeout=20, tag=None):
        """Create a new connection to IBKR with tracking."""
        if client_id is None:
            # Generate a unique client ID for each connection
            client_id = int(time.time() % 9000) + 1000 + self.connection_count
            
        # Create a unique tag if not provided
        if tag is None:
            tag = f"conn_{self.connection_count}"
            
        # Create the connection
        ib = IB()
        
        try:
            logger.info(f"Connecting to IBKR at {host}:{port} with client ID {client_id} (tag: {tag})")
            ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout)
            
            if ib.isConnected():
                # Track the connection
                self.connection_count += 1
                self.active_connections[tag] = {
                    'ib': ib,
                    'client_id': client_id,
                    'created_at': datetime.now(),
                    'request_count': 0
                }
                
                # Register with monitor
                self.monitor.register_connection(tag, client_id)
                
                logger.info(f"✓ Successfully connected to IBKR ({tag})")
                return ib
            else:
                tqdm.write(f"❌ Failed to connect to IBKR ({tag})")
                return None
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {str(e)}")
            tqdm.write(f"❌ Error connecting to IBKR: {str(e)}")
            if ib:
                try:
                    ib.disconnect()
                except:
                    pass
            return None
            
    def disconnect(self, ib=None, tag=None):
        """Disconnect a specific connection by IB instance or tag."""
        if tag and tag in self.active_connections:
            # Disconnect by tag
            conn_info = self.active_connections[tag]
            ib_to_disconnect = conn_info['ib']
            
            try:
                if ib_to_disconnect and ib_to_disconnect.isConnected():
                    ib_to_disconnect.disconnect()
                    logger.info(f"✓ Disconnected from IBKR ({tag})")
            except Exception as e:
                logger.error(f"Error disconnecting from IBKR ({tag}): {str(e)}")
                
            # Unregister from monitor
            self.monitor.unregister_connection(tag)
            
            # Remove from active connections
            del self.active_connections[tag]
            return True
            
        elif ib:
            # Disconnect by IB instance
            # Find the tag for this connection
            for t, conn_info in list(self.active_connections.items()):
                if conn_info['ib'] == ib:
                    try:
                        if ib.isConnected():
                            ib.disconnect()
                            logger.info(f"✓ Disconnected from IBKR ({t})")
                    except Exception as e:
                        logger.error(f"Error disconnecting from IBKR ({t}): {str(e)}")
                    
                    # Unregister from monitor
                    self.monitor.unregister_connection(t)
                    
                    # Remove from active connections
                    del self.active_connections[t]
                    return True
        
        # If we get here, no matching connection was found
        return False
        
    def disconnect_all(self):
        """Disconnect all active connections."""
        for tag, conn_info in list(self.active_connections.items()):
            ib = conn_info['ib']
            try:
                if ib and ib.isConnected():
                    ib.disconnect()
                    logger.info(f"✓ Disconnected from IBKR ({tag})")
            except Exception as e:
                logger.error(f"Error disconnecting from IBKR ({tag}): {str(e)}")
            
            # Unregister from monitor
            self.monitor.unregister_connection(tag)
                
        # Clear the connections dictionary
        self.active_connections = {}
        
    def get_active_connection_count(self):
        """Return the number of active connections."""
        return len(self.active_connections)
        
    def get_connection_info(self):
        """Return information about all active connections."""
        info = {}
        for tag, conn_info in self.active_connections.items():
            info[tag] = {
                'client_id': conn_info['client_id'],
                'created_at': conn_info['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                'request_count': conn_info['request_count'],
                'connected': conn_info['ib'].isConnected() if conn_info['ib'] else False
            }
        return info
        
    def increment_request_count(self, ib):
        """Increment the request count for a connection."""
        for tag, conn_info in self.active_connections.items():
            if conn_info['ib'] == ib:
                conn_info['request_count'] += 1
                # Log request in monitor
                self.monitor.log_request(tag)
                return conn_info['request_count']
        return 0
    
    def log_metrics(self):
        """Log connection metrics."""
        return self.monitor.log_metrics()
        
    def __del__(self):
        """Ensure all connections are closed and log metrics when the object is garbage collected."""
        try:
            # Log final metrics
            self.monitor.log_metrics(level=logging.INFO)
            # Disconnect all connections
            self.disconnect_all()
        except:
            pass

connection_manager = IBKRConnectionManager()












def connect_to_ibkr(host='127.0.0.1', port=7497, client_id=None, timeout=20, batch_tag=None):
    """Connect to Interactive Brokers TWS or Gateway using the connection manager."""
    global connection_manager
    
    # Create a tag for this connection if not provided
    if batch_tag is None:
        batch_tag = f"batch_{int(time.time())}"
        
    # Use the connection manager to create a connection
    ib = connection_manager.connect(host, port, client_id, timeout, batch_tag)
    
    # Only show connections log if count is at a significant threshold (e.g., 2, 5, 10)
    if ib:
        active_count = connection_manager.get_active_connection_count()
        # Only log if count is 1 (first connection) or reaches specific thresholds
        if active_count == 1 or active_count == 2 or active_count == 5 or active_count % 10 == 0:
            logger.info(f"ℹ️ Active connections: {active_count}")
        
    return ib

def get_contract(ticker, security_type='STK', exchange='SMART', currency='USD'):
    """Create an IBKR contract object for the specified ticker."""
    return Contract(symbol=ticker, secType=security_type, exchange=exchange, currency=currency)

async def download_historical_data(ib, ticker, data_type, bar_size, duration_str, use_rth=True):
    """Download historical data for a ticker with specified type and bar size."""
    global connection_manager
    try:
        # Create contract
        contract = get_contract(ticker)
        
        # Update connection metrics (increment request count)
       
        connection_manager.increment_request_count(ib)
        
        # Request contract details to verify the contract
        contract_details = await ib.reqContractDetailsAsync(contract)
        if not contract_details:
            logger.warning(f"No contract details found for {ticker}")
            return None
        
        logger.info(f"Downloading {duration_str} of {bar_size} {data_type} data for {ticker}")
        
        # Increment request count again for the historical data request
        connection_manager.increment_request_count(ib)
        
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
                         delay=REQUEST_DELAY, connection=None, pbar=None, max_retries=3):
    """Process a single ticker with improved connection handling and retry logic."""
    global connection_manager
    # Create result variables
    daily_dataframes = {}
    minute_dataframes = {}
    success = False
    need_to_disconnect = False
    connection_tag = None
    
    # Track retries for session connection issues
    retry_count = 0
    
    while retry_count <= max_retries:
        # Connect to IBKR if a connection wasn't provided
        ib = connection
        if not ib:
            # Create a unique tag for this connection
            connection_tag = f"ticker_{ticker}_{int(time.time())}"
            ib = connect_to_ibkr(host=host, port=port, batch_tag=connection_tag)
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
                connection_manager.increment_request_count(ib)
            
            # Process all data types concurrently
            session_error = False
            for data_type, task in tasks:
                try:
                    df = await task
                    if df is not None and not df.empty:
                        daily_dataframes[data_type] = df
                    await asyncio.sleep(delay)  # Small delay between tasks
                except Exception as e:
                    error_msg = str(e)
                    if "Trading TWS session is connected from a different IP address" in error_msg:
                        session_error = True
                        logger.warning(f"TWS session IP error for {ticker}. Will retry in 60 seconds. Retry {retry_count+1}/{max_retries+1}")
                        break
                    else:
                        logger.error(f"Error processing {ticker} daily {data_type}: {str(e)}")
            
            # If we encountered a session error, wait and retry
            if session_error:
                if need_to_disconnect and connection_tag:
                    connection_manager.disconnect(tag=connection_tag)
                    need_to_disconnect = False
                
                retry_count += 1
                if retry_count <= max_retries:
                    # Wait for 60 seconds before retrying
                    await asyncio.sleep(60)
                    continue
                else:
                    logger.error(f"Max retries reached for {ticker}. Giving up.")
                    return False
            
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
                            
                            # Update progress bar if provided
                            if pbar and success:
                                pbar.update(1)
                                # Update statistics in the postfix
                                current_fails = int(pbar.postfix.split('failed=')[1].split(',')[0]) if hasattr(pbar, 'postfix') and 'failed=' in str(pbar.postfix) else 0
                                success_rate = (pbar.n / (pbar.n + current_fails)) * 100 if (pbar.n + current_fails) > 0 else 0
                                pbar.set_postfix(failed=current_fails, success_rate=f"{success_rate:.1f}%", last=ticker)
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
            
            # If we reach here without errors, we're done with this ticker
            break
                
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Check if this was a session error that we should retry
            if "Trading TWS session is connected from a different IP address" in str(e):
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(f"Will retry {ticker} in 60 seconds. Retry {retry_count}/{max_retries+1}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    continue
            
            success = False
            break  # Exit retry loop on non-session-related errors
        
        finally:
            # Disconnect only if we created the connection ourselves
            if need_to_disconnect and connection_tag:
                connection_manager.disconnect(tag=connection_tag)
                logger.info(f"Disconnected from IBKR for {ticker}")
            
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
                      quiet_mode=False, pbar=None, max_retries=3):
    """Process a batch of tickers with improved connection handling."""
    global connection_manager
    if not tickers:
        return 0, 0
    
    # Create a batch tag for tracking this connection
    batch_tag = f"batch_{int(time.time())}"
    
    # Create IBKR connection for this batch
    logger.info(f"\nConnecting to IBKR for batch of {len(tickers)} tickers...")
    ib = connect_to_ibkr(host=host, port=port, batch_tag=batch_tag)
    
    if not ib:
        tqdm.write("❌ Failed to connect to IBKR for batch processing.")
        return 0, len(tickers)
    
    success_count = 0
    fail_count = 0
    
    try:
        # Process tickers with limited concurrency
        tasks = []
        for ticker in tickers:
            task = asyncio.create_task(process_ticker(
                ticker=ticker,
                daily_duration=daily_duration,
                minute_duration=minute_duration,
                include_minute_data=include_minute_data,
                use_rth=use_rth,
                connection=ib,  # Pass the shared connection
                pbar=pbar,      # Pass the progress bar
                max_retries=max_retries  # Pass the max retries parameter
            ))
            tasks.append((task, ticker))
        
        # Process each ticker and count successes/failures
        for task, ticker in tasks:
            try:
                result = await task
                if result:
                    success_count += 1
                    # Progress bar is updated inside process_ticker
                else:
                    fail_count += 1
                    # Update failed count in progress bar
                    if pbar:
                        current_fails = int(pbar.postfix.split('failed=')[1].split(',')[0]) if hasattr(pbar, 'postfix') and 'failed=' in str(pbar.postfix) else 0
                        new_fails = current_fails + 1
                        success_rate = (pbar.n / (pbar.n + new_fails)) * 100 if (pbar.n + new_fails) > 0 else 0
                        pbar.set_postfix(failed=new_fails, success_rate=f"{success_rate:.1f}%", last=ticker)
            except Exception as e:
                fail_count += 1
                logger.error(f"Error processing {ticker}: {str(e)}")
                tqdm.write(f"❌ Error processing {ticker}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        tqdm.write(f"❌ Error in batch processing: {str(e)}")
        fail_count += len(tickers) - success_count
    
    finally:
        # Ensure we disconnect using the connection manager
        connection_manager.disconnect(tag=batch_tag)
        logger.info(f"✓ Disconnected from IBKR (batch {batch_tag})")
        
        # Log remaining active connections
        active_count = connection_manager.get_active_connection_count()
        if active_count > 0:
            tqdm.write(f"⚠️ Warning: Still have {active_count} active connections")
        
        return success_count, fail_count

async def process_all_tickers(tickers, host='127.0.0.1', port=7497, batch_size=DEFAULT_BATCH_SIZE, 
                             daily_duration=DAILY_DURATION, minute_duration=MINUTE_DURATION,
                             include_minute_data=False, use_rth=True, resume=False,
                             max_workers=MAX_WORKERS_PER_BATCH, quiet_mode=False):
    """Process all tickers with a cleaner, more responsive progress display."""
    # Import os_module to avoid shadowing
    import os as os_module
    import shutil
    
    processed_tickers = []
    failed_tickers = []
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
    
    # Clear terminal for clean display if not in quiet mode
    if not quiet_mode:
        os_module.system('cls' if os_module.name == 'nt' else 'clear')
        
        # Show header with summary information
        print("="*80)
        print("IBKR STOCK DATA DOWNLOADER - RUNNING")
        print("="*80)
        print(f"Tickers: {len(remaining_tickers)} pending, {len(processed_tickers)} completed")
        print(f"Batch size: {batch_size}, Workers: {max_workers}")
        print(f"Minute data: {'Skipped' if not include_minute_data else 'Included'}")
        print("="*80)
    
    # Get terminal width for the progress bar
    try:
        # Try to get terminal size using shutil
        terminal_width = shutil.get_terminal_size().columns
    except Exception:
        try:
            # Try to get terminal size using os_module
            terminal_width = os_module.get_terminal_size().columns
        except Exception:
            # Default fallback
            terminal_width = 100
    
    # Create A SINGLE progress bar for ALL tickers (not per batch) with adaptive width
    pbar = tqdm(total=len(remaining_tickers), desc="Download Progress", 
                unit="ticker", ncols=terminal_width, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    # Process tickers in batches, but update progress bar per ticker
    for i in range(0, len(remaining_tickers), batch_size):
        batch = remaining_tickers[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(remaining_tickers) + batch_size - 1) // batch_size
        
        # Update progress bar description
        pbar.set_description(f"Batch {batch_num}/{total_batches}")
        
        # Process the batch - note we pass the main progress bar
        batch_success, batch_fail = await process_batch(
            tickers=batch,
            host=host,
            port=port,
            daily_duration=daily_duration,
            minute_duration=minute_duration,
            include_minute_data=include_minute_data,
            use_rth=use_rth,
            max_workers=max_workers,
            quiet_mode=quiet_mode,
            pbar=pbar  # Pass the progress bar for per-ticker updates
        )
        
        # Track failed tickers for recovery
        failed_batch_tickers = batch[batch_success:]  # Approximates failed tickers
        failed_tickers.extend(failed_batch_tickers)
        
        # These are now redundant as individual tickers update the bar
        # but kept for tracking the counts
        success_count += batch_success
        fail_count += batch_fail
        
        # Update processed and remaining tickers lists
        batch_processed = batch[:batch_success]  # This is approximate
        processed_tickers.extend(batch_processed)
        remaining_after_batch = remaining_tickers[i+batch_size:]
        
        # Save progress after each batch without disrupting progress bar
        save_progress(processed_tickers, total_count, remaining_after_batch, pbar=pbar)
        
        # Print batch completion info without disrupting progress bar
        logger.info(f"Batch {batch_num}/{total_batches} completed: "
              f"{batch_success}/{len(batch)} successful, {batch_fail} failed")
        
        # Pause between batches
        if batch_num < total_batches:
            logger.info(f"Pausing for {BATCH_PAUSE} seconds before next batch...")
            await asyncio.sleep(BATCH_PAUSE)
    
    # Complete the progress bar
    pbar.close()
    
    # After processing all batches, try to recover any failed tickers
    if failed_tickers:
        tqdm.write("\n" + "="*80)
        tqdm.write(f"Recovery pass: Attempting to download {len(failed_tickers)} failed tickers")
        tqdm.write("="*80)
        
        # Create a new progress bar for the recovery pass
        recovery_pbar = tqdm(total=len(failed_tickers), desc="Recovery Pass", 
                unit="ticker", ncols=terminal_width, position=0, leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        recovery_success = 0
        recovery_failed = 0
        
        # Let's wait a bit before starting recovery, in case this was due to a temporary session issue
        tqdm.write("Waiting 2 minutes before starting recovery...")
        await asyncio.sleep(120)
        
        # Process failed tickers in smaller batches for recovery
        recovery_batch_size = min(5, batch_size)
        for i in range(0, len(failed_tickers), recovery_batch_size):
            recovery_batch = failed_tickers[i:i+recovery_batch_size]
            batch_num = (i // recovery_batch_size) + 1
            total_batches = (len(failed_tickers) + recovery_batch_size - 1) // recovery_batch_size
            
            # Update progress bar description
            recovery_pbar.set_description(f"Recovery Batch {batch_num}/{total_batches}")
            
            # Process the batch with longer delays
            recovery_batch_success, recovery_batch_fail = await process_batch(
                tickers=recovery_batch,
                host=host,
                port=port,
                daily_duration=daily_duration,
                minute_duration=minute_duration,
                include_minute_data=include_minute_data,
                use_rth=use_rth,
                max_workers=max(1, max_workers//2),  # Use fewer workers for recovery
                quiet_mode=quiet_mode,
                pbar=recovery_pbar
            )
            
            recovery_success += recovery_batch_success
            recovery_failed += recovery_batch_fail
            
            # Update processed tickers list with recovered tickers
            recovery_batch_processed = recovery_batch[:recovery_batch_success]
            processed_tickers.extend(recovery_batch_processed)
            
            # Longer pause between recovery batches
            if batch_num < total_batches:
                tqdm.write(f"Pausing for {BATCH_PAUSE*2} seconds before next recovery batch...")
                await asyncio.sleep(BATCH_PAUSE * 2)
        
        # Close the recovery progress bar
        recovery_pbar.close()
        
        # Update the overall success/fail counts
        success_count += recovery_success
        fail_count = fail_count - recovery_success + recovery_failed
        
        # Log recovery results
        tqdm.write("\n" + "="*80)
        tqdm.write("RECOVERY RESULTS")
        tqdm.write("="*80)
        tqdm.write(f"Attempted recovery of {len(failed_tickers)} tickers")
        tqdm.write(f"Successfully recovered: {recovery_success} tickers ({(recovery_success/len(failed_tickers))*100:.1f}%)")
        tqdm.write(f"Still failed: {recovery_failed} tickers")
        tqdm.write("="*80)
    
    # Final summary
    tqdm.write("\n" + "="*80)
    tqdm.write("DOWNLOAD COMPLETE")
    tqdm.write("="*80)
    tqdm.write(f"Total processed: {success_count + fail_count} tickers")
    tqdm.write(f"Successful: {success_count} ({success_count/(success_count+fail_count)*100:.1f}%)")
    tqdm.write(f"Failed: {fail_count}")
    tqdm.write("="*80)
    
    return success_count, fail_count, total_count




def display_header(args):
    """Display a clean header with configuration information."""
    # Clear terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display header
    print("="*80)
    print("IBKR STOCK DATA DOWNLOADER")
    print("="*80)
    print(f"Python Version: {os.sys.version.split()[0]}")
    
    # Configuration info
    config_lines = [
        f"Host: {args.host}, Port: {args.port}",
        f"Batch size: {args.batch_size}, Workers: {args.max_workers}",
        f"Delays: Request={args.request_delay}s, Batch pause={args.batch_pause}s",
        f"Data: Daily={args.daily_duration}, Minute={'Skipped' if args.skip_minute_data else args.minute_duration}",
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



def main():
    # Create argument parser
    global connection_manager
    
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
    parser.add_argument('--minute-duration', type=str, default=MINUTE_DURATION, help=f'Minute data duration (default: {MINUTE_DURATION})')
    parser.add_argument('--skip-minute-data', action='store_true', help='Skip downloading minute data')
    parser.add_argument('--use-rth', action='store_true', help='Use regular trading hours only')
    parser.add_argument('--trades-only', action='store_true', help='Download only TRADES data (OHLCV), no BID/ASK/MIDPOINT')
    
    # Performance settings
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Number of tickers to process in each batch (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS_PER_BATCH, help=f'Maximum number of concurrent workers per batch (default: {MAX_WORKERS_PER_BATCH})')
    parser.add_argument('--request-delay', type=float, default=REQUEST_DELAY, help=f'Delay between requests in seconds (default: {REQUEST_DELAY})')
    parser.add_argument('--batch-pause', type=int, default=BATCH_PAUSE, help=f'Pause between batches in seconds (default: {BATCH_PAUSE})')
    
    # Display settings
    parser.add_argument('--quiet', action='store_true', help='Disable the mid-batch progress bar, showing only the overall progress')
    
    args = parser.parse_args()
    
    # After parsing args
    if args.trades_only:
        globals()['DATA_TYPES'] = ['TRADES']

    # Update global variables via globals() dictionary
    globals()['REQUEST_DELAY'] = args.request_delay
    globals()['BATCH_PAUSE'] = args.batch_pause
    
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
    
    try:
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
        
        # Log connection metrics
        print("\nLogging connection metrics...")
        
        connection_manager.log_metrics()
    
    finally:
        # Ensure all connections are closed when the program exits
        
        active_connections = connection_manager.get_active_connection_count()
        if active_connections > 0:
            print(f"Cleaning up {active_connections} active connections...")
            # This will also log final metrics
            connection_manager.disconnect_all()
            print("All connections closed.")





if __name__ == "__main__":
    main()