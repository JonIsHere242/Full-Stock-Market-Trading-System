#!/usr/bin/env python
"""
Util.py - Unified trading system utilities
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import traceback
import math
import warnings
from pytz import timezone as pytz_timezone
import sys 
import inspect
import hashlib
from pathlib import Path
import threading
import atexit
import platform
import subprocess
import argparse



SIGNALS_FILE = 'Data/Production/LiveTradingData/pending_signals.parquet'
POSITIONS_FILE = 'Data/Production/LiveTradingData/active_positions.parquet'
COMPLETED_TRADES_FILE = 'Data/Production/LiveTradingData/completed_trades.parquet'
# Constant file paths for trading data
BACKTESTER_PARQUET = '_Buy_Signals.parquet'
LIVE_TRADER_PARQUET = '_Live_trades.parquet'

# Define schema for parquet files
PARQUET_SCHEMA = {
    'Symbol': 'string',
    'LastBuySignalDate': 'datetime64[ns]',
    'LastBuySignalPrice': 'float64',
    'IsCurrentlyBought': 'bool',
    'ConsecutiveLosses': 'int64',
    'LastTradedDate': 'datetime64[ns]',
    'UpProbability': 'float64',
    'LastSellPrice': 'float64',
    'PositionSize': 'float64'
}




# Strategy parameters in tuple format for backtrader
STRATEGY_PARAMS_TUPLE = (
    # Signal generation parameters
    ('up_prob_threshold', 0.60),        # Probability threshold for buy signals
    ('up_prob_min_trigger', 0.70),      # Minimum probability to trigger buy
    
    # Position management parameters     
    ('max_positions', 4),               # Maximum concurrent positions
    ('reserve_percent', 0.10),          # Cash reserve percentage
    ('max_group_allocation', 0.20),     # Maximum allocation to a group
    
    # Risk management parameters   
    ('risk_per_trade_pct', 3.0),        # Risk per trade as percentage
    ('max_position_pct', 20.0),         # Maximum position size as percentage 
    ('min_position_pct', 5.0),          # Minimum position size as percentage
    ('atr_period', 14),                 # ATR calculation period
    
    # Stop loss and take profit parameters
    ('stop_loss_atr_multiple', 0.75),   # Stop loss ATR multiplier
    ('trailing_stop_atr_multiple', 2.0), # Trailing stop ATR multiplier
    ('take_profit_percent', 20.0),      # Take profit threshold percentage
    
    # Position timeout and evaluation parameters         
    ('position_timeout', 5),            # Maximum days to hold a position
    ('min_daily_return', 1.0),          # Minimum expected daily return
    
    # Other system parameters
    ('lockup_days', 3),                 # Trading lockup period
    ('rule_201_threshold', -9.99),      # Rule 201 threshold
    ('rule_201_cooldown', 1),           # Rule 201 cooldown period
    ('stop_loss_percent', 5.0),         # Standard stop loss percentage
    ('expected_profit_per_day_percentage', 0.25) # Expected profit per day
)

# Strategy parameters as dictionary for easier access
STRATEGY_PARAMS = {
    # Signal generation parameters
    'up_prob_threshold': 0.60,          # Probability threshold for buy signals  
    'up_prob_min_trigger': 0.70,        # Minimum probability to trigger buy
    
    # Position management parameters 
    'max_positions': 4,                 # Maximum concurrent positions
    'reserve_percent': 0.10,            # Cash reserve percentage
    'max_group_allocation': 0.20,       # Maximum allocation to a group
    
    # Risk management parameters
    'risk_per_trade_pct': 3.0,          # Risk per trade as percentage
    'max_position_pct': 20.0,           # Maximum position size as percentage
    'min_position_pct': 5.0,            # Minimum position size as percentage
    'atr_period': 14,                   # ATR calculation period
    
    # Stop loss and take profit parameters
    'stop_loss_atr_multiple': 0.75,     # Stop loss ATR multiplier 
    'trailing_stop_atr_multiple': 2.0,  # Trailing stop ATR multiplier
    'take_profit_percent': 20.0,        # Take profit threshold percentage
    
    # Position timeout and evaluation parameters
    'position_timeout': 5,              # Maximum days to hold a position
    'min_daily_return': 1.0,            # Minimum expected daily return
    
    # Other system parameters
    'lockup_days': 3,                   # Trading lockup period
    'rule_201_threshold': -9.99,        # Rule 201 threshold
    'rule_201_cooldown': 1,             # Rule 201 cooldown period
    'stop_loss_percent': 5.0,           # Standard stop loss percentage
    'expected_profit_per_day_percentage': 0.25 # Expected profit per day
}



#atr take profit 1.0

#atr stop loss 1.0

##change the atr trailing stop each 




def should_use_production_files():
    """Check if we should use the new production files"""
    all_exist = (
        os.path.exists(SIGNALS_FILE) and
        os.path.exists(POSITIONS_FILE)
    )
    
    if all_exist:
        # Check if signals file contains data for today's date
        try:
            signals_df = pd.read_parquet(SIGNALS_FILE)
            if not signals_df.empty:
                tomorrow = get_next_trading_day(datetime.now().date())
                for_tomorrow = signals_df['TargetDate'].dt.date == tomorrow
                return for_tomorrow.any()
        except Exception as e:
            logging.error(f"Error reading signals file: {str(e)}")
    
    return False





class Colors:
    """ANSI color codes for colored terminal output"""
    RESET = "\033[0m"
    INFO = "\033[38;2;100;149;237m" # Cornflower blue
    WARN = "\033[38;2;220;220;0m"     # Yellow
    ERROR = "\033[38;2;220;0;0m"      # Red
    DETAIL = "\033[38;2;0;200;0m"     # Green
    DEBUG = "\033[38;2;0;180;180m"    # Cyan
    SUCCESS = "\033[38;2;50;220;50m"  # Bright Green
    TRACE = "\033[38;2;180;180;180m"  # Gray
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    TIMESTAMP = "\033[38;2;150;150;150m" # Gray

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""
    COLOR_MAP = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARN,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.ERROR,
    }

    def format(self, record):
        # Get color from extra or level
        color = getattr(record, 'color', self.COLOR_MAP.get(record.levelno, Colors.RESET))
        
        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)
        colored_ts = f"{Colors.TIMESTAMP}{timestamp}{Colors.RESET}"
        
        # Format level name
        levelname = f"{color}{record.levelname}{Colors.RESET}"
        
        # Include file and line info in debug mode
        file_info = ""
        if record.levelno <= logging.DEBUG:
            filename = os.path.basename(record.pathname)
            file_info = f"{Colors.TRACE}[{filename}:{record.lineno}]{Colors.RESET} "
        
        # Build formatted message
        return f"{colored_ts} - {levelname} - {file_info}{record.getMessage()}"

# Global registry to track initialized loggers
_LOGGER_REGISTRY = {}
_LOG_LOCK = threading.RLock()

def get_script_name():
    """Get the name of the calling script, handling both direct execution and imports"""
    # Start from one frame up to skip this function
    for frame in inspect.stack()[1:]:
        module = inspect.getmodule(frame[0])
        # Skip frames from this module
        if module and module.__name__ != __name__:
            # Get the file path
            file_path = Path(frame.filename)
            # If it's a .py file, return its stem
            if file_path.suffix.lower() == '.py':
                return file_path.stem
            # For Jupyter notebooks, create a stable name
            elif file_path.suffix.lower() == '.ipynb':
                # Hash the full path to create a stable identifier
                hash_obj = hashlib.md5(str(file_path).encode())
                return f"jupyter_{hash_obj.hexdigest()[:8]}"
    
    # Fallback to the main script name
    return Path(sys.argv[0]).stem if sys.argv[0] else "unknown"

def setup_logging(log_dir='Data/logging', console=True, debug=False, script_name=None):
    """
    Set up logging with consistent configuration.
    
    Args:
        log_dir: Directory for log files
        console: Whether to output to console
        debug: Whether to enable debug mode
        script_name: Override script name detection (optional)
        
    Returns:
        The configured logger
    """
    with _LOG_LOCK:
        # Determine the script name if not provided
        if script_name is None:
            script_name = get_script_name()
        
        # Check if this script already has a logger
        if script_name in _LOGGER_REGISTRY:
            return _LOGGER_REGISTRY[script_name]
        
        # Create a unique logger for this script
        logger = logging.getLogger(script_name)
        # Clear any existing handlers
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # Set up the log file path
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{script_name}.log"
        
        # File handler - one per script
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler with colors
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
            console_handler.setFormatter(ColoredFormatter())
            logger.addHandler(console_handler)
        
        # Set the logger's level
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Store in registry
        _LOGGER_REGISTRY[script_name] = logger
        
        # Register cleanup on exit
        atexit.register(lambda: logger.handlers.clear())
        
        # Log initialization
        logger.info(f"Logging initialized for {script_name}")
        if debug:
            logger.debug("Debug mode enabled")
        
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        if hasattr(pd, 'errors') and hasattr(pd.errors, 'PerformanceWarning'):
            warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        
        return logger

def get_logger(script_name=None, **kwargs):
    """
    Get or create a logger for the specified script name
    
    Args:
        script_name: Optional script name override
        **kwargs: Additional arguments for setup_logging if a new logger is created
        
    Returns:
        The configured logger
    """
    if script_name is None:
        script_name = get_script_name()
    
    if script_name in _LOGGER_REGISTRY:
        return _LOGGER_REGISTRY[script_name]
    
    return setup_logging(script_name=script_name, **kwargs)

def dprint(message, level="INFO", show_timestamp=True, indent=0, logger=None):
    """
    Enhanced debug print with colors and logging integration
    
    Args:
        message: The message to print
        level: One of "INFO", "WARN", "ERROR", "DETAIL", "DEBUG", "SUCCESS", "TRACE"
        show_timestamp: Whether to include timestamp
        indent: Number of spaces to indent
        logger: Optional logger instance to also log the message
    """
    color = getattr(Colors, level, Colors.INFO)
    level_str = f"[{level}]".ljust(8)
    indent_str = " " * indent

    # Console output
    if show_timestamp:
        timestamp = f"{Colors.TIMESTAMP}{datetime.now().strftime('%H:%M:%S.%f')[:-3]}{Colors.RESET} "
    else:
        timestamp = ""
    
    print(f"{timestamp}{indent_str}{color}{level_str}{Colors.RESET} {message}")

    # File logging - try to get logger if not provided
    if logger is None and _LOGGER_REGISTRY:
        script_name = get_script_name()
        if script_name in _LOGGER_REGISTRY:
            logger = _LOGGER_REGISTRY[script_name]
    
    if logger:
        level_map = {
            'INFO': logging.INFO,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
            'DEBUG': logging.DEBUG,
            'SUCCESS': logging.INFO,
            'DETAIL': logging.DEBUG,
            'TRACE': logging.DEBUG
        }
        log_level = level_map.get(level, logging.INFO)
        logger.log(log_level, f"{indent_str}{message}", extra={'color': color})

class LogPerformance:
    """Context manager for timing and logging performance metrics"""
    
    def __init__(self, operation_name, logger=None, level="INFO"):
        self.operation_name = operation_name
        self.logger = logger if logger else get_logger()
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(
            getattr(logging, self.level) if hasattr(logging, self.level) else logging.INFO,
            f"Starting: {self.operation_name}"
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type:
            self.logger.error(f"Failed: {self.operation_name} after {duration} - {exc_val}")
        else:
            self.logger.log(
                getattr(logging, self.level) if hasattr(logging, self.level) else logging.INFO,
                f"Completed: {self.operation_name} in {duration}"
            )

def log_progress(iterable, logger=None, every=None, total=None, description="Processing"):
    """
    Log progress through an iterable
    
    Args:
        iterable: The iterable to process
        logger: Logger to use (will auto-detect if None)
        every: Log every N items (will auto-calculate if None)
        total: Total items (will use len() if None and available)
        description: Description to include in log messages
        
    Returns:
        Generator yielding items from the original iterable
    """
    if logger is None:
        logger = get_logger()
    
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None
    
    if every is None:
        if total and total > 0:
            # Log approximately 10 times during the process
            every = max(1, total // 10)
        else:
            every = 100
    
    start_time = datetime.now()
    last_log_time = start_time
    
    for i, item in enumerate(iterable, 1):
        yield item
        
        if i % every == 0 or (total and i == total):
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            # Only log if it's been at least 0.5 seconds since last log
            if (current_time - last_log_time).total_seconds() >= 0.5:
                if total:
                    percent = (i / total) * 100
                    msg = f"{description}: {i}/{total} ({percent:.1f}%) in {elapsed}"
                    
                    # Estimate time remaining
                    if i > 0:
                        items_per_sec = i / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                        if items_per_sec > 0:
                            remaining_items = total - i
                            est_remaining_sec = remaining_items / items_per_sec
                            est_completion = current_time + timedelta(seconds=est_remaining_sec)
                            msg += f", est. completion at {est_completion.strftime('%H:%M:%S')}"
                else:
                    msg = f"{description}: {i} items in {elapsed}"
                
                logger.info(msg)
                last_log_time = current_time

# Configure main logger with standard settings
def configure_main_logger(debug=False):
    """Configure the main script logger with standard settings"""
    return setup_logging(
        log_dir='Data/logging',
        console=True,
        debug=debug
    )

##=================================================[Terminal logger for quick retrieval]=================================================##







def cache_terminal(line_count=100, label=None, cache_dir='Data/logging/cache'):
    """
    Cache terminal output to a file using clipboard
    
    Args:
        line_count: Number of lines to capture (default 100)
        label: Optional label to identify this cache entry
        cache_dir: Directory to save cache files
        
    Returns:
        Path to the cached file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_str = f"_{label}" if label else ""
    filename = f"terminal_cache_{timestamp}{label_str}.log"
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Full path to the cache file
    cache_file = cache_path / filename
    
    # Header for the cache file
    header = f"""
==============================================================
= TERMINAL OUTPUT CACHE - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
= Captured {line_count} lines{f' - {label}' if label else ''}
= System: {platform.system()} {platform.release()}
= Python: {sys.version.split()[0]}
==============================================================

"""
    
    # Try to use clipboard content (requires manual copy)
    try:
        # Check if pyperclip or pywin32 is available
        clipboard_content = None
        try:
            import pyperclip
            clipboard_content = pyperclip.paste()
        except ImportError:
            try:
                # Use win32clipboard if available; ensure pywin32 is installed: pip install pywin32
                import win32clipboard
                win32clipboard.OpenClipboard()
                if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_TEXT):
                    clipboard_content = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT).decode('utf-8')
                win32clipboard.CloseClipboard()
            except ImportError:
                clipboard_content = None
        
        if clipboard_content and len(clipboard_content) > 10:
            # Split by lines and take the last 'line_count' lines
            lines = clipboard_content.split('\n')
            if len(lines) > line_count:
                lines = lines[-line_count:]
            terminal_output = '\n'.join(lines)
            
            # Write to the cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(terminal_output)
            
            print(f"Cached {len(lines)} lines of terminal output from clipboard to {cache_file}")
            print("Note: For best results, select and copy (Ctrl+A, Ctrl+C) terminal content before running this command")
            return str(cache_file)
    except Exception as e:
        print(f"Error using clipboard: {str(e)}")
    
    # Fallback: just capture command history
    terminal_output = ""
    try:
        # Get command history
        history_cmd = f"Get-History -Count {line_count} | Format-Table Id, CommandLine -AutoSize | Out-String"
        history_result = subprocess.run(
            ["powershell", "-Command", history_cmd],
            capture_output=True, text=True, encoding='utf-8'
        )
        terminal_output = "=== COMMAND HISTORY ===\n\n" + history_result.stdout
        
        # Add system info
        sys_cmd = """
        Get-Process | Sort-Object -Property CPU -Descending | Select-Object -First 5 | 
        Format-Table -Property Name, CPU, WorkingSet -AutoSize | Out-String
        """
        sys_result = subprocess.run(
            ["powershell", "-Command", sys_cmd],
            capture_output=True, text=True, encoding='utf-8'
        )
        terminal_output += "\n=== SYSTEM INFORMATION ===\n\n" + sys_result.stdout
        
        # Add message to instruct on better capture
        terminal_output += "\n=== NOTE ===\n"
        terminal_output += "For better terminal capture:\n"
        terminal_output += "1. Install pyperclip package: pip install pyperclip\n"
        terminal_output += "2. Select all terminal text (Ctrl+A)\n"
        terminal_output += "3. Copy to clipboard (Ctrl+C)\n"
        terminal_output += "4. Run this command again"
        
    except Exception as e:
        terminal_output = f"Error capturing command history: {str(e)}"
    
    # Write to the cache file
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(terminal_output)
    
    print(f"Cached terminal output to {cache_file}")
    print("Note: For best results, install pyperclip (pip install pyperclip)")
    print("      Then select and copy (Ctrl+A, Ctrl+C) terminal content before running this command")
    
    return str(cache_file)







def show_cache_entries(count=5, cache_dir='Data/logging/cache'):
    """
    Show the most recent terminal cache entries
    
    Args:
        count: Number of recent entries to show
        cache_dir: Directory with cache files
        
    Returns:
        List of recent cache files
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist.")
        return []
    
    # Get all cache files and sort by modification time (newest first)
    cache_files = list(cache_path.glob("terminal_cache_*.log"))
    cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Show the most recent entries
    print(f"\nRecent terminal cache entries ({min(count, len(cache_files))} of {len(cache_files)}):")
    for i, file in enumerate(cache_files[:count]):
        size_kb = file.stat().st_size / 1024
        mod_time = datetime.datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        # Try to extract the label if present
        label = file.stem.split('_', 3)[-1] if len(file.stem.split('_')) > 3 else ""
        label_str = f" - {label}" if label else ""
        print(f"{i+1}. {file.name}{label_str} ({size_kb:.1f} KB, {mod_time})")
    
    return cache_files[:count]

def read_cache(cache_file=None, index=None, cache_dir='Data/logging/cache'):
    """
    Read and display a cached terminal output file
    
    Args:
        cache_file: Path to the cache file to read
        index: Index of the recent cache file to read (1-based)
        cache_dir: Directory with cache files
        
    Returns:
        Content of the cache file
    """
    if cache_file is None and index is None:
        # If no arguments, show recent files and prompt for index
        files = show_cache_entries(cache_dir=cache_dir)
        if not files:
            return "No cache files found."
        
        try:
            index = int(input("\nEnter number to read (or press Enter to cancel): "))
            if index < 1 or index > len(files):
                return "Invalid index."
            cache_file = files[index-1]
        except ValueError:
            return "Operation cancelled."
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif index is not None:
        # Get file by index
        files = show_cache_entries(count=index, cache_dir=cache_dir)
        if not files or index > len(files):
            return f"No cache file found at index {index}."
        cache_file = files[index-1]
    
    # Read the cache file
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n===== Content of {Path(cache_file).name} =====")
        print(content)
        print("="*40)
        return content
    except Exception as e:
        return f"Error reading cache file: {str(e)}"

##=================================================[Trading Functions used in backtester and live trader]=================================================##

def get_last_trading_date():
    """Get the last trading date from NYSE calendar."""
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now().date()
    
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    
    if schedule.empty:
        raise Exception("No trading days found in the past 10 days.")
    
    if today in schedule.index.date:
        today_market_open = schedule.loc[schedule.index.date == today, 'market_open'].iloc[0]
        
        if today_market_open.tzinfo is None:
            today_market_open = today_market_open.replace(tzinfo=pytz_timezone('UTC'))
        
        now_utc = datetime.now(pytz_timezone('UTC'))
        
        if now_utc < today_market_open:
            schedule = schedule[schedule.index.date < today]
    
    if not schedule.empty:
        last_trading_date = schedule.index[-1].date()
        return last_trading_date
    else:
        raise Exception("No trading days found in the past 10 days after excluding today.")

def get_previous_trading_day(current_date, days_back=1):
    """Get the nth previous trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    end_date = current_date.date()
    start_date = end_date - timedelta(days=days_back * 2)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    valid_days = schedule[schedule.index.date <= end_date]
    if len(valid_days) < days_back:
        raise ValueError(f"Not enough trading days found before {end_date}")
    
    return valid_days.index[-days_back].date()

def get_next_trading_day(current_date):
    """Get the next trading day."""
    nyse = mcal.get_calendar('NYSE')
    current_date = pd.Timestamp(current_date)
    
    start_date = current_date.date() + timedelta(days=1)
    end_date = start_date + timedelta(days=10)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    if schedule.empty:
        raise ValueError(f"No trading days found after {start_date}")
    
    return schedule.index[0].date()

def is_market_open():
    """Check if the market is currently open."""
    nyse = mcal.get_calendar('NYSE')
    now = pd.Timestamp.now(tz='America/New_York')
    today_date = now.date()
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today_date, end_date=today_date)
    if schedule.empty:
        return False, "Not a trading day"
    
    market_open = schedule.iloc[0]['market_open'].tz_convert('America/New_York')
    market_close = schedule.iloc[0]['market_close'].tz_convert('America/New_York')
    
    if now < market_open:
        return False, "Market not yet open"
    elif now > market_close:
        return False, "Market closed for the day"
    else:
        return True, "Market open"

def read_trading_data(file_path=None, is_live=False):
    """Read the trading data from parquet file."""
    if file_path is None:
        file_path = LIVE_TRADER_PARQUET if is_live else BACKTESTER_PARQUET
    
    try:
        if not os.path.exists(file_path):
            # Create a new DataFrame with the right schema
            df = pd.DataFrame(columns=list(PARQUET_SCHEMA.keys()))
            
            # Set proper dtypes
            for col, dtype in PARQUET_SCHEMA.items():
                if dtype == 'datetime64[ns]':
                    df[col] = pd.Series(dtype='datetime64[ns]')
                else:
                    df[col] = pd.Series(dtype=dtype)
            
            # Save the empty DataFrame
            df.to_parquet(file_path, index=False)
            logging.info(f"Created new empty trading data file: {file_path}")
            return df
        
        # Read existing file
        df = pd.read_parquet(file_path)
        
        # Ensure all expected columns exist
        for col, dtype in PARQUET_SCHEMA.items():
            if col not in df.columns:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.Series(dtype='datetime64[ns]')
                elif dtype == 'float64':
                    df[col] = pd.Series(dtype='float64')
                elif dtype == 'int64':
                    df[col] = pd.Series(dtype='int64')
                elif dtype == 'bool':
                    df[col] = pd.Series(dtype='bool')
                elif dtype == 'string':
                    df[col] = pd.Series(dtype='string')
        
        # Fix data types as needed
        for col, dtype in PARQUET_SCHEMA.items():
            if col in df.columns:
                if dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                elif dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'bool':
                    if df[col].dtype != 'bool':
                        df[col] = df[col].astype('bool')
                elif dtype == 'string':
                    if df[col].dtype != 'string':
                        df[col] = df[col].astype('string')
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading trading data from {file_path}: {str(e)}")
        logging.error(traceback.format_exc())
        
        # Return an empty DataFrame as a fallback
        return pd.DataFrame(columns=list(PARQUET_SCHEMA.keys()))

def write_trading_data(df, file_path=None, is_live=False):
    """Write the trading data to parquet file."""
    if file_path is None:
        file_path = LIVE_TRADER_PARQUET if is_live else BACKTESTER_PARQUET
    
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure correct data types
        for col, dtype in PARQUET_SCHEMA.items():
            if col in df.columns:
                if dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                elif dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'bool':
                    if df[col].dtype != 'bool':
                        df[col] = df[col].astype('bool')
                elif dtype == 'string':
                    if df[col].dtype != 'string':
                        df[col] = df[col].astype('string')
                elif dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert datetime columns to handle null values properly
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in datetime_cols:
            df[col] = df[col].astype('object').where(df[col].notnull(), None)
        
        # Write to parquet
        df.to_parquet(file_path, index=False)
        logging.info(f"Successfully wrote trading data to {file_path}")
        
    except Exception as e:
        logging.error(f"Error writing trading data to {file_path}: {str(e)}")
        logging.error(traceback.format_exc())

def update_trade_data(symbol, trade_type, price=None, date=None, position_size=None, is_live=False):
    """
    Update trading data for a specific symbol
    
    Parameters:
    - symbol: Stock symbol
    - trade_type: 'buy' or 'sell'
    - price: Trade price
    - date: Trade date
    - position_size: Position size
    - is_live: Whether to update the live trading data
    """
    try:
        if date is None:
            date = datetime.now().date()
        
        df = read_trading_data(is_live=is_live)
        
        if trade_type.lower() == 'buy':
            if symbol in df['Symbol'].values:
                df.loc[df['Symbol'] == symbol, 'LastBuySignalPrice'] = price
                df.loc[df['Symbol'] == symbol, 'LastBuySignalDate'] = pd.Timestamp(date)
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
                if position_size is not None:
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = position_size
            else:
                new_row = {
                    'Symbol': symbol,
                    'LastBuySignalDate': pd.Timestamp(date),
                    'LastBuySignalPrice': price,
                    'IsCurrentlyBought': True,
                    'ConsecutiveLosses': 0,
                    'LastTradedDate': pd.NaT,
                    'UpProbability': 0.0,
                    'LastSellPrice': float('nan'),
                    'PositionSize': position_size if position_size is not None else float('nan')
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        elif trade_type.lower() == 'sell':
            if symbol in df['Symbol'].values:
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(date)
                if price is not None:
                    df.loc[df['Symbol'] == symbol, 'LastSellPrice'] = price
                df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
        
        write_trading_data(df, is_live=is_live)
        logging.info(f"Updated trade data for {symbol}: {trade_type.capitalize()} at {price}")
        
    except Exception as e:
        logging.error(f"Error updating trade data for {symbol}: {str(e)}")
        logging.error(traceback.format_exc())



def get_buy_signals(is_live=False):
    """Get potential buy signals from trading data."""
    try:
        if should_use_production_files():
            logging.info("Using new production files for buy signals")
            df = pd.read_parquet(SIGNALS_FILE)
            
            # Get tomorrow's date
            tomorrow = get_next_trading_day(datetime.now().date())
            
            # Filter signals for tomorrow
            signals = df[df['TargetDate'].dt.date == tomorrow]
            
            if not signals.empty:
                result = signals.to_dict('records')
                logging.info(f"Found {len(result)} signals for tomorrow in production files")
                return result
            else:
                logging.info("No signals found for tomorrow in production files")
        
        # Fall back to old method if needed
        logging.info("Using legacy files for buy signals")
        df = read_trading_data(is_live=is_live)
        
        # Get signals that are not currently bought
        buy_signals = df[
            (df['IsCurrentlyBought'] == False) & 
            (df['LastBuySignalDate'].notna())
        ]
        
        # Check if we have the LastBuySignalDate column and it's not empty
        if 'LastBuySignalDate' in buy_signals.columns and not buy_signals.empty:
            # Get last trading date
            last_date = get_last_trading_date()
            
            # Filter for signals on or after the previous trading day
            prev_day = get_previous_trading_day(last_date)
            recent_signals = buy_signals[buy_signals['LastBuySignalDate'].dt.date >= prev_day]
            
            if not recent_signals.empty:
                result = recent_signals.to_dict('records')
                logging.info(f"Found {len(result)} recent buy signals in legacy files")
                return result
            else:
                logging.info("No recent buy signals found in legacy files")
                return buy_signals.to_dict('records')  # Return all signals if no recent ones
            
        return buy_signals.to_dict('records')
        
    except Exception as e:
        logging.error(f"Error getting buy signals: {str(e)}")
        logging.error(traceback.format_exc())
        return []








def should_sell(current_price, entry_price, entry_date, current_date, 
               stop_loss_percent, take_profit_percent, position_timeout, 
               expected_profit_per_day_percentage, verbose=False):
    """
    Determine if a position should be sold based on various criteria.
    
    Parameters:
    - current_price: Current price of the security
    - entry_price: Entry price of the position
    - entry_date: Date when the position was entered
    - current_date: Current date
    - stop_loss_percent: Stop-loss percentage
    - take_profit_percent: Take-profit percentage
    - position_timeout: Maximum number of days to hold the position
    - expected_profit_per_day_percentage: Expected minimum profit per day
    - verbose: Whether to print detailed information
    
    Returns:
    - Boolean indicating whether to sell the position and the reason
    """
    try:
        # Calculate metrics
        profit_pct = ((current_price / entry_price) - 1) * 100
        days_held = (current_date - entry_date).days
        
        # Calculate stop-loss and take-profit thresholds
        stop_loss_threshold = entry_price * (1 - stop_loss_percent / 100)
        take_profit_threshold = entry_price * (1 + take_profit_percent / 100)
        
        # Calculate minimum expected profit
        min_expected_profit = days_held * expected_profit_per_day_percentage
        
        # Check conditions
        stop_loss_triggered = current_price <= stop_loss_threshold
        take_profit_triggered = current_price >= take_profit_threshold
        timeout_triggered = days_held >= position_timeout
        poor_performance = days_held > 2 and profit_pct < min_expected_profit
        
        # Log details if verbose
        if verbose:
            logging.info(f"Sell Analysis - Profit: {profit_pct:.2f}%, Days Held: {days_held}")
            logging.info(f"Stop Loss: {stop_loss_triggered} (Threshold: {stop_loss_threshold:.2f})")
            logging.info(f"Take Profit: {take_profit_triggered} (Threshold: {take_profit_threshold:.2f})")
            logging.info(f"Timeout: {timeout_triggered} (Max: {position_timeout} days)")
            logging.info(f"Performance: {poor_performance} (Min Expected: {min_expected_profit:.2f}%)")
        
        # Determine if any sell condition is met
        should_sell_flag = stop_loss_triggered or take_profit_triggered or timeout_triggered or poor_performance
        reason = None
        
        if should_sell_flag:
            if stop_loss_triggered:
                reason = "Stop Loss"
            elif take_profit_triggered:
                reason = "Take Profit"
            elif timeout_triggered:
                reason = "Position Timeout"
            elif poor_performance:
                reason = "Poor Performance"
            
            if verbose:
                logging.info(f"Sell signal triggered: {reason}")
        
        return should_sell_flag, reason
    
    except Exception as e:
        logging.error(f"Error in should_sell: {str(e)}")
        return False, "Error"

def sync_trading_data():
    """Synchronize data between backtester and live trader."""
    try:
        # Read both datasets
        backtest_data = read_trading_data(is_live=False)
        live_data = read_trading_data(is_live=True)
        
        # 1. Find new buy signals from backtester
        backtest_buy_signals = backtest_data[
            (backtest_data['IsCurrentlyBought'] == False) & 
            (backtest_data['LastBuySignalDate'].notna())
        ]
        
        for _, signal in backtest_buy_signals.iterrows():
            symbol = signal['Symbol']
            
            # Check if symbol exists in live data
            if symbol in live_data['Symbol'].values:
                # Update only if backtester signal is newer
                live_signal = live_data[live_data['Symbol'] == symbol].iloc[0]
                if (pd.isna(live_signal['LastBuySignalDate']) or 
                    (not pd.isna(signal['LastBuySignalDate']) and 
                     signal['LastBuySignalDate'] > live_signal['LastBuySignalDate'])):
                    
                    logging.info(f"Updating buy signal for {symbol} from backtester")
                    live_data.loc[live_data['Symbol'] == symbol, 'LastBuySignalDate'] = signal['LastBuySignalDate']
                    live_data.loc[live_data['Symbol'] == symbol, 'LastBuySignalPrice'] = signal['LastBuySignalPrice']
                    live_data.loc[live_data['Symbol'] == symbol, 'UpProbability'] = signal['UpProbability']
            else:
                # Add new symbol to live data
                logging.info(f"Adding new buy signal for {symbol} from backtester")
                new_row = signal.to_dict()
                live_data = pd.concat([live_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # 2. Update backtester with position status from live trader
        live_positions = live_data[live_data['IsCurrentlyBought'] == True]
        
        for _, position in live_positions.iterrows():
            symbol = position['Symbol']
            
            # Update backtester data with position status
            if symbol in backtest_data['Symbol'].values:
                logging.info(f"Updating backtester position status for {symbol}")
                backtest_data.loc[backtest_data['Symbol'] == symbol, 'IsCurrentlyBought'] = True
                backtest_data.loc[backtest_data['Symbol'] == symbol, 'PositionSize'] = position['PositionSize']
                
                # If position was sold, update sell information
                if pd.notna(position['LastTradedDate']) and pd.notna(position['LastSellPrice']):
                    backtest_data.loc[backtest_data['Symbol'] == symbol, 'LastTradedDate'] = position['LastTradedDate']
                    backtest_data.loc[backtest_data['Symbol'] == symbol, 'LastSellPrice'] = position['LastSellPrice']
        
        # 3. Update closed positions in backtester
        live_closed = live_data[
            (live_data['IsCurrentlyBought'] == False) & 
            (live_data['LastTradedDate'].notna())
        ]
        
        for _, closed_position in live_closed.iterrows():
            symbol = closed_position['Symbol']
            
            # Update backtester data with closed position info
            if symbol in backtest_data['Symbol'].values:
                if backtest_data.loc[backtest_data['Symbol'] == symbol, 'IsCurrentlyBought'].iloc[0]:
                    logging.info(f"Updating backtester with closed position for {symbol}")
                    backtest_data.loc[backtest_data['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                    backtest_data.loc[backtest_data['Symbol'] == symbol, 'LastTradedDate'] = closed_position['LastTradedDate']
                    backtest_data.loc[backtest_data['Symbol'] == symbol, 'LastSellPrice'] = closed_position['LastSellPrice']
        
        # Write updated data back
        write_trading_data(backtest_data, is_live=False)
        write_trading_data(live_data, is_live=True)
        
        logging.info("Successfully synchronized trading data between backtester and live trader")
        return True
        
    except Exception as e:
        logging.error(f"Error synchronizing trading data: {str(e)}")
        logging.error(traceback.format_exc())
        return False

##=================================================[More extensive trade logging and risk metrics]=================================================##

def colorize_output(value, label, good_threshold, bad_threshold, lower_is_better=False, reverse=False, unicorn_multiplier=20.0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{label:<30}\033[38;2;150;150;150mN/A        \033[0m[\033[38;2;150;150;150mNo Data\033[0m]"
    
    def get_color_code(normalized_value, is_unicorn=False):
        if is_unicorn:
            return "\033[38;2;100;149;237m"  # Cornflower blue
        colors = [
            (0, 235, 0),    # Bright Green
            (0, 180, 0),    # Normal Green
            (220, 220, 0),  # Yellow
            (220, 140, 0),  # Orange
            (220, 0, 0),    # Red
            (240, 0, 0)     # Bright Red
        ]
        
        index = min(int(normalized_value * (len(colors) - 1)), len(colors) - 2)
        
        t = (normalized_value * (len(colors) - 1)) - index
        r = int(colors[index][0] * (1-t) + colors[index+1][0] * t)
        g = int(colors[index][1] * (1-t) + colors[index+1][1] * t)
        b = int(colors[index][2] * (1-t) + colors[index+1][2] * t)
        return f"\033[38;2;{r};{g};{b}m"
    
    def get_quality_tag(normalized_value, is_unicorn=False):
        if is_unicorn:
            return "Unicorn 🦄"
            
        if normalized_value <= 0.15:
            return "Excellent"
        elif normalized_value <= 0.3:
            return "Very Good"
        elif normalized_value <= 0.45:
            return "Good"
        elif normalized_value <= 0.6:
            return "Average"
        elif normalized_value <= 0.75:
            return "Below Average"
        elif normalized_value <= 0.9:
            return "Poor"
        else:
            return "Unacceptable"

    is_unicorn = False
    if not lower_is_better and value >= good_threshold * unicorn_multiplier:
        is_unicorn = True
    elif lower_is_better and value <= good_threshold / unicorn_multiplier:
        is_unicorn = True

    if reverse:
        good_threshold, bad_threshold = bad_threshold, good_threshold

    try:
        if is_unicorn:
            normalized_value = 0  # Best value
        elif lower_is_better:
            if value <= good_threshold:
                normalized_value = 0  # Best value (Green)
            elif value >= bad_threshold:
                normalized_value = 1  # Worst value (Red)
            else:
                normalized_value = (value - good_threshold) / (bad_threshold - good_threshold)
        else:
            if value >= good_threshold:
                normalized_value = 0  # Best value (Green)
            elif value <= bad_threshold:
                normalized_value = 1  # Worst value (Red)
            else:
                normalized_value = (good_threshold - value) / (good_threshold - bad_threshold)
    except Exception as e:
        return f"{label:<30}\033[38;2;150;150;150mError      \033[0m[\033[38;2;150;150;150mCalculation Error\033[0m]"

    color_code = get_color_code(normalized_value, is_unicorn)
    quality_tag = get_quality_tag(normalized_value, is_unicorn)
    
    if isinstance(value, float):
        value_str = f"{value:.2f}"
    else:
        value_str = str(value)
    
    return f"{label:<30}{color_code}{value_str:<10}\033[0m[{color_code}{quality_tag}\033[0m]"

def calculate_enhanced_metrics(results, daily_returns=None, trade_results=None):
    """
    Calculate enhanced trading metrics that better represent stock sniping strategies
    
    Parameters:
    results (dict): Original results dictionary containing standard metrics
    daily_returns (list): List of daily return percentages (optional)
    trade_results (list): List of individual trade P&L values (optional)
    
    Returns:
    dict: Enhanced metrics
    """
    enhanced = {}
    
    # Clone existing SQN for reference
    enhanced['original_sqn'] = results.get('sqn_value', 0)
    
    # 1. Modified SQN (normalize by percentage returns rather than dollar amounts)
    if trade_results is not None and results.get('initial_value', 0) > 0:
        # Convert dollar P&L to percentage returns
        pct_returns = [r / results['initial_value'] * 100 for r in trade_results]
        if len(pct_returns) > 1:
            mean_r = np.mean(pct_returns)
            std_dev = np.std(pct_returns)
            if std_dev > 0:
                enhanced['modified_sqn'] = (mean_r / std_dev) * math.sqrt(len(pct_returns))
            else:
                enhanced['modified_sqn'] = float('inf')
        else:
            enhanced['modified_sqn'] = 0
    else:
        enhanced['modified_sqn'] = 0
    
    # Determine modified SQN quality
    enhanced['modified_sqn_quality'] = get_sqn_quality(enhanced['modified_sqn'])
    
    # 2. MAR Ratio (CAGR / Max DD)
    if results.get('max_dd', 0) > 0:
        enhanced['mar_ratio'] = results.get('annualized_return', 0) / results.get('max_dd', 100)
    else:
        enhanced['mar_ratio'] = float('inf')
    
    # 3. K-Ratio (consistency of returns)
    if daily_returns is not None and len(daily_returns) > 20:
        # Generate equity curve from daily returns
        equity_curve = 100 * (1 + np.array(daily_returns) / 100).cumprod()
        
        # Linear regression of equity curve
        x = np.arange(len(equity_curve))
        slope, _ = np.polyfit(x, np.log(equity_curve), 1)
        
        # Standard error of linear regression
        y_fitted = np.exp(slope * x)
        std_error = np.sqrt(np.sum((equity_curve - y_fitted) ** 2) / (len(equity_curve) - 2))
        
        if std_error > 0:
            enhanced['k_ratio'] = slope * 100 / std_error
        else:
            enhanced['k_ratio'] = float('inf')
    else:
        enhanced['k_ratio'] = 0
    
    # 4. Expectancy Score
    win_rate = results.get('percent_profitable', 0) / 100
    avg_win = results.get('avg_win_pct', 0)
    avg_loss = results.get('avg_loss_pct', 0)
    enhanced['expectancy_score'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # 5. Profit Per Day of Capital at Risk
    avg_risk_per_trade = min(results.get('avg_loss_pct', 1), 1)  # Default to 1% if not available
    if avg_risk_per_trade > 0:
        enhanced['profit_per_risk_day'] = results.get('profit_per_day', 0) / (results.get('initial_value', 1) * avg_risk_per_trade / 100)
    else:
        enhanced['profit_per_risk_day'] = 0
    
    # 6. Information Ratio (assumes a benchmark return of 10% annually)
    benchmark_return = 10  # Standard market benchmark annual return
    if results.get('annualized_volatility', 0) > 0:
        excess_return = results.get('annualized_return', 0) - benchmark_return
        enhanced['information_ratio'] = excess_return / results.get('annualized_volatility', 1)
    else:
        enhanced['information_ratio'] = 0
    
    # 7. Serenity Ratio (Uses downside deviation instead of standard deviation)
    if daily_returns is not None:
        downside_returns = [r for r in daily_returns if r < 0]
        if downside_returns:
            downside_dev = np.std(downside_returns) * np.sqrt(252 / len(downside_returns))
            if downside_dev > 0:
                enhanced['serenity_ratio'] = results.get('annualized_return', 0) / downside_dev
            else:
                enhanced['serenity_ratio'] = float('inf')
        else:
            enhanced['serenity_ratio'] = float('inf')
    else:
        enhanced['serenity_ratio'] = 0
    
    # 8. Trade Efficiency (How much of potential profit is captured)
    if results.get('mfe_avg', 0) > 0:
        enhanced['trade_efficiency'] = results.get('avg_profit_per_trade', 0) / results.get('mfe_avg', 1) * 100
    else:
        enhanced['trade_efficiency'] = 0
    
    # 9. Risk-Adjusted CAGR
    # This adjusts the CAGR for the underlying volatility
    cagr = results.get('annualized_return', 0)
    volatility = results.get('annualized_volatility', 0)
    if volatility > 0:
        enhanced['risk_adjusted_cagr'] = cagr / volatility
    else:
        enhanced['risk_adjusted_cagr'] = cagr
    
    # 10. System Robustness Score (composite score from multiple metrics)
    # This is a custom metric weighted towards what's important for sniping strategies
    robustness_components = [
        min(1, results.get('total_closed', 0) / 80) * 0.10,  # Trade count component (increased weight)
        min(1, results.get('profit_factor', 0) / 3) * 0.25,  # Profit factor component (increased weight)
        min(1, enhanced.get('expectancy_score', 0) / 2) * 0.15,  # Expectancy component
        min(1, max(0, 3 - results.get('max_dd', 0) / 10) / 3) * 0.20,  # Drawdown resilience (increased weight)
        min(1, max(0, results.get('positive_days_pct', 0)) / 50) * 0.10,  # Consistency
        min(1, enhanced.get('modified_sqn', 0) / 4) * 0.20,  # Modified SQN
    ]
    
    # Initialize with partial components that we have
    valid_components = [c for c in robustness_components if not math.isnan(c)]
    if valid_components:
        enhanced['system_robustness_score'] = sum(valid_components) / sum(c is not None for c in robustness_components)
    else:
        enhanced['system_robustness_score'] = 0
    
    # 11. Position Sizing Kelly Adjustment
    # This is a safer Kelly criterion for position sizing (half-Kelly)
    original_kelly = results.get('kelly_percentage', 0)
    enhanced['half_kelly'] = original_kelly / 2
    enhanced['quarter_kelly'] = original_kelly / 4
    
    # 12. Maximum Favorable Excursion to Maximum Adverse Excursion ratio
    if results.get('mae_avg', 0) > 0:
        enhanced['mfe_mae_ratio'] = results.get('mfe_avg', 0) / results.get('mae_avg', 1)
    else:
        enhanced['mfe_mae_ratio'] = float('inf')
    
    # 13. Stress Test Score - how well the system handles consecutive losses
    max_cons_losses = results.get('max_consecutive_losses', 0)
    avg_loss_pct = results.get('avg_loss_pct', 1)
    recovery_factor = results.get('recovery_factor', 1)
    
    # Calculate the drawdown from max consecutive losses
    stress_dd = (1 - (1 - avg_loss_pct/100) ** max_cons_losses) * 100
    
    # Higher score is better (100 = excellent, 0 = poor)
    if stress_dd > 0:
        enhanced['stress_test_score'] = min(100, 100 * (recovery_factor / stress_dd) ** 0.5)
    else:
        enhanced['stress_test_score'] = 100
    
    return enhanced

def get_sqn_quality(sqn_value):
    """Determine the SQN quality description based on the SQN value."""
    sqn_descriptions = {
        (float('-inf'), 0): "Negative",
        (0, 1.6): "Poor",
        (1.6, 2.0): "Below Average",
        (2.0, 2.5): "Average",
        (2.5, 3.0): "Good",
        (3.0, 5.0): "Excellent",
        (5.0, 7.0): "Superb",
        (7.0, float('inf')): "Holy Grail Potential"
    }
    
    for (low, high), desc in sqn_descriptions.items():
        if low <= sqn_value < high:
            return desc
    
    return "Unknown"

def extract_trade_results(results):
    """
    Extract trade results from the results dictionary
    
    This function would normally parse your actual trade history,
    but for this example we'll simulate trades based on the provided metrics
    """
    trade_results = []
    
    # Simulate winning trades
    won_total = results.get('won_total', 0)
    won_avg = results.get('won_avg', 0)
    won_max = results.get('won_max', 0)
    
    # Simulate losing trades
    lost_total = results.get('lost_total', 0)
    lost_avg = results.get('lost_avg', 0)
    lost_max = results.get('lost_max', 0)
    
    # Create a distribution of winning trades (log-normal distribution)
    if won_total > 0 and won_avg > 0:
        # Parameters for log-normal distribution
        sigma = 0.8  # Controls the spread
        mu = np.log(won_avg) - sigma**2/2  # Ensure mean is won_avg
        
        # Generate winning trades
        win_trades = np.random.lognormal(mu, sigma, won_total)
        
        # Scale to ensure the maximum win is close to won_max
        if won_max > 0:
            scale_factor = won_max / max(win_trades)
            win_trades = win_trades * scale_factor
            
            # Ensure average is still close to won_avg
            avg_adjustment = won_avg / np.mean(win_trades)
            win_trades = win_trades * avg_adjustment
            
        trade_results.extend(win_trades)
    
    # Create a distribution of losing trades (log-normal for magnitude, then negative)
    if lost_total > 0 and lost_avg > 0:
        # Parameters for log-normal distribution
        sigma = 0.7  # Less variability in losses typically
        mu = np.log(lost_avg) - sigma**2/2  # Ensure mean is lost_avg
        
        # Generate losing trades
        loss_trades = np.random.lognormal(mu, sigma, lost_total)
        
        # Scale to ensure the maximum loss is close to lost_max
        if lost_max > 0:
            scale_factor = lost_max / max(loss_trades)
            loss_trades = loss_trades * scale_factor
            
            # Ensure average is still close to lost_avg
            avg_adjustment = lost_avg / np.mean(loss_trades)
            loss_trades = loss_trades * avg_adjustment
            
        # Convert to negative values
        loss_trades = -loss_trades
        
        trade_results.extend(loss_trades)
    
    # Shuffle the trades to interleave wins and losses (more realistic)
    np.random.shuffle(trade_results)
    
    return trade_results

def extract_daily_returns(results):
    """
    Extract or simulate daily returns based on the provided metrics
    """
    daily_returns = []
    
    # Get key metrics from results
    annualized_return = results.get('annualized_return', 0) / 100  # Convert to decimal
    annualized_volatility = results.get('annualized_volatility', 0) / 100  # Convert to decimal
    daily_volatility = results.get('daily_volatility', 0) / 100  # Convert to decimal
    positive_days_pct = results.get('positive_days_pct', 0) / 100  # Convert to decimal
    day_count = results.get('day_count', 252)
    
    # Generate daily returns with appropriate volatility
    if day_count > 0 and daily_volatility > 0:
        # Generate random normal returns
        daily_returns = np.random.normal(0, daily_volatility, day_count)
        
        # Adjust to match the desired annualized return
        daily_mean = annualized_return / 252  # Approximate daily mean return
        daily_returns = daily_returns + daily_mean
        
        # Adjust to match the positive days percentage
        if positive_days_pct > 0:
            # Sort returns and determine threshold for positive/negative split
            sorted_returns = np.sort(daily_returns)
            threshold_index = int((1 - positive_days_pct) * len(sorted_returns))
            if threshold_index < len(sorted_returns):
                threshold = sorted_returns[threshold_index]
                
                # Adjust returns around threshold to match positive_days_pct
                for i in range(len(daily_returns)):
                    if i <= threshold_index and daily_returns[i] > threshold:
                        daily_returns[i] = threshold * 0.99
                    elif i > threshold_index and daily_returns[i] < threshold:
                        daily_returns[i] = threshold * 1.01
        
        # Convert to percentage for consistency with other metrics
        daily_returns = daily_returns * 100
    
    return daily_returns

def print_enhanced_metrics(results, enhanced_metrics):
    """Print enhanced metrics section with colorized output"""
    print("\n" + "=" * 80)
    print(" Enhanced Stock Sniper Metrics ".center(80))
    print("=" * 80)
    
    # SQN Comparison
    print("\nSQN Normalization Metrics:")
    print(colorize_output(enhanced_metrics['original_sqn'], "Original SQN:", 3.0, 1.6))
    print(colorize_output(enhanced_metrics['modified_sqn'], "Modified SQN (% normalized):", 3.0, 1.6))
    modified_sqn_quality = enhanced_metrics['modified_sqn_quality']
    print(f"{'Modified SQN Quality:':<30}{get_color_for_sqn(modified_sqn_quality)}{modified_sqn_quality:<10}\033[0m")
    
    # Risk-Reward Metrics
    print("\nRisk-Reward Metrics:")
    print(colorize_output(enhanced_metrics['mar_ratio'], "MAR Ratio:", 2.0, 0.5))
    print(colorize_output(enhanced_metrics['expectancy_score'], "Expectancy Score:", 1.0, 0.2))
    print(colorize_output(enhanced_metrics['profit_per_risk_day'], "Profit per Risk-Day:", 0.2, 0))
    print(colorize_output(enhanced_metrics['information_ratio'], "Information Ratio:", 0.5, 0))
    print(colorize_output(enhanced_metrics['serenity_ratio'], "Serenity Ratio:", 1.0, 0.2))
    print(colorize_output(enhanced_metrics['mfe_mae_ratio'], "MFE/MAE Ratio:", 2.0, 1.0))
    
    # Strategy Consistency Metrics
    print("\nStrategy Consistency Metrics:")
    print(colorize_output(enhanced_metrics['k_ratio'], "K-Ratio:", 1.0, 0.3))
    print(colorize_output(enhanced_metrics['trade_efficiency'], "Trade Efficiency %:", 50, 20))
    print(colorize_output(enhanced_metrics['risk_adjusted_cagr'], "Risk-Adjusted CAGR:", 1.0, 0.5))
    print(colorize_output(enhanced_metrics['stress_test_score'], "Stress Test Score:", 70, 30))
    
    # Position Sizing Recommendations
    print("\nPosition Sizing Recommendations:")
    print(colorize_output(results.get('kelly_percentage', 0), "Full Kelly %:", 20, 5))
    print(colorize_output(enhanced_metrics['half_kelly'], "Half Kelly % (Recommended):", 15, 2.5))
    print(colorize_output(enhanced_metrics['quarter_kelly'], "Quarter Kelly % (Conservative):", 10, 1))
    
    # Overall System Quality
    print("\nOverall System Quality:")
    print(colorize_output(enhanced_metrics['system_robustness_score'], "System Robustness Score:", 0.7, 0.3))
    
    # Print a summary interpretation
    print("\nSystem Interpretation:")
    print_system_interpretation(results, enhanced_metrics)
    
def print_enhanced_consistency_metrics(results):
    """Print enhanced strategy consistency metrics if available."""
    # Check if we have enhanced metrics
    has_enhanced = any(key.startswith('enhanced_') for key in results.keys())
    
    if has_enhanced:
        print("\nStrategy Consistency Metrics:")
        if 'enhanced_k_ratio' in results:
            print(colorize_output(results['enhanced_k_ratio'], "K-Ratio:", 1.0, 0.3))
        if 'enhanced_trade_efficiency' in results:
            print(colorize_output(results['enhanced_trade_efficiency'], "Trade Efficiency %:", 50, 20))
        if 'enhanced_risk_adjusted_cagr' in results:
            print(colorize_output(results['enhanced_risk_adjusted_cagr'], "Risk-Adjusted CAGR:", 1.0, 0.5))
        if 'enhanced_stress_test_score' in results:
            print(colorize_output(results['enhanced_stress_test_score'], "Stress Test Score:", 70, 30))
        if 'enhanced_system_robustness_score' in results:
            print(colorize_output(results['enhanced_system_robustness_score'], "System Robustness Score:", 0.7, 0.3))
        if 'enhanced_serenity_ratio' in results:
            print(colorize_output(results['enhanced_serenity_ratio'], "Serenity Ratio:", 1.0, 0.2))
        if 'enhanced_mfe_mae_ratio' in results:
            print(colorize_output(results['enhanced_mfe_mae_ratio'], "MFE/MAE Ratio:", 2.0, 1.0))
        if 'enhanced_mar_ratio' in results:
            print(colorize_output(results['enhanced_mar_ratio'], "MAR Ratio:", 2.0, 0.5))
        if 'enhanced_information_ratio' in results:
            print(colorize_output(results['enhanced_information_ratio'], "Information Ratio:", 0.5, 0))
        if 'enhanced_expectancy_score' in results:
            print(colorize_output(results['enhanced_expectancy_score'], "Expectancy Score:", 1.0, 0.2))
        if 'enhanced_profit_per_risk_day' in results:
            print(colorize_output(results['enhanced_profit_per_risk_day'], "Profit per Risk-Day:", 0.2, 0))

def print_position_sizing_recommendations(results):
    """Print position sizing recommendations if available."""
    if 'enhanced_half_kelly' in results:
        print("\nPosition Sizing Recommendations:")
        print(colorize_output(results.get('kelly_percentage', 0), "Full Kelly %:", 20, 5))
        print(colorize_output(results['enhanced_half_kelly'], "Half Kelly % (Recommended):", 15, 2.5))
        print(colorize_output(results['enhanced_quarter_kelly'], "Quarter Kelly % (Conservative):", 10, 1))

def get_color_for_sqn(sqn_quality):
    """Get color code for SQN quality text."""
    color_map = {
        "Holy Grail Potential": "\033[38;2;100;149;237m",  # Cornflower blue (Unicorn)
        "Superb": "\033[38;2;0;235;0m",                   # Bright Green
        "Excellent": "\033[38;2;0;180;0m",                # Normal Green
        "Good": "\033[38;2;145;200;20m",                  # Light Green
        "Average": "\033[38;2;220;220;0m",                # Yellow
        "Below Average": "\033[38;2;220;140;0m",          # Orange
        "Poor": "\033[38;2;220;0;0m",                     # Red
        "Negative": "\033[38;2;240;0;0m",                 # Bright Red
        "Unknown": "\033[38;2;150;150;150m"               # Gray
    }
    return color_map.get(sqn_quality, "\033[38;2;150;150;150m")

def print_system_interpretation(results, enhanced_metrics=None):
    """Print an interpretation of the trading system based on the enhanced metrics."""
    # If enhanced_metrics is not provided but exists in results with 'enhanced_' prefix, use those
    if enhanced_metrics is None:
        enhanced_metrics = {}
        for key in results:
            if key.startswith('enhanced_'):
                enhanced_metrics[key[9:]] = results[key]  # Remove 'enhanced_' prefix
    
    print("\nSystem Interpretation:")
    
    # Determine system strengths
    strengths = []
    if enhanced_metrics.get('modified_sqn', results.get('enhanced_modified_sqn', 0)) > 3.0:
        strengths.append("Strong edge with normalized position sizing")
    if results.get('profit_factor', 0) > 2.5:
        strengths.append("Excellent profit factor")
    if enhanced_metrics.get('expectancy_score', results.get('enhanced_expectancy_score', 0)) > 1.0:
        strengths.append("High trade expectancy")
    if enhanced_metrics.get('risk_adjusted_cagr', results.get('enhanced_risk_adjusted_cagr', 0)) > 1.0:
        strengths.append("Good risk-adjusted returns")
    if enhanced_metrics.get('system_robustness_score', results.get('enhanced_system_robustness_score', 0)) > 0.6:
        strengths.append("Robust overall system")
    if enhanced_metrics.get('stress_test_score', results.get('enhanced_stress_test_score', 0)) > 70:
        strengths.append("Strong performance under stress")
    
    # Determine system weaknesses
    weaknesses = []
    if enhanced_metrics.get('modified_sqn', results.get('enhanced_modified_sqn', 0)) < 1.6:
        weaknesses.append("Weak edge with normalized position sizing")
    if results.get('max_consecutive_losses', 0) > 5:
        weaknesses.append("Vulnerable to consecutive losses")
    if enhanced_metrics.get('trade_efficiency', results.get('enhanced_trade_efficiency', 0)) < 30:
        weaknesses.append("Low trade execution efficiency")
    if results.get('positive_days_pct', 0) < 35:
        weaknesses.append("Low percentage of positive days")
    if enhanced_metrics.get('system_robustness_score', results.get('enhanced_system_robustness_score', 0)) < 0.4:
        weaknesses.append("System may lack robustness")
    
    # Print system strengths
    if strengths:
        print("System Strengths:")
        for i, strength in enumerate(strengths[:3], 1):  # Limit to top 3
            print(f"  {i}. \033[38;2;0;180;0m{strength}\033[0m")
    
    # Print system weaknesses
    if weaknesses:
        print("\nAreas for Improvement:")
        for i, weakness in enumerate(weaknesses[:3], 1):  # Limit to top 3
            print(f"  {i}. \033[38;2;220;140;0m{weakness}\033[0m")
    
    # Print recommendations
    print("\nRecommendations:")
    modified_sqn = enhanced_metrics.get('modified_sqn', results.get('enhanced_modified_sqn', 0))
    original_sqn = enhanced_metrics.get('original_sqn', results.get('sqn_value', 0))
    
    if modified_sqn > original_sqn:
        print("  • \033[38;2;255;255;255mUse Modified SQN for system quality evaluation\033[0m")
    
    half_kelly = enhanced_metrics.get('half_kelly', results.get('enhanced_half_kelly', 0))
    if half_kelly > 0:
        print(f"  • \033[38;2;255;255;255mConsider position sizing at {half_kelly:.1f}% (Half Kelly)\033[0m")
    
    serenity_ratio = enhanced_metrics.get('serenity_ratio', results.get('enhanced_serenity_ratio', 0))
    mar_ratio = enhanced_metrics.get('mar_ratio', results.get('enhanced_mar_ratio', 0))
    if serenity_ratio > mar_ratio:
        print("  • \033[38;2;255;255;255mFocus on Serenity Ratio rather than traditional risk metrics\033[0m")
    
    if results.get('max_consecutive_losses', 0) > 5:
        print("  • \033[38;2;255;255;255mImplement drawdown controls to limit consecutive losses\033[0m")

def integrate_enhanced_metrics_into_results(results):
    """
    Calculate enhanced metrics and add them to results dictionary
    without disrupting the existing print flow
    """
    # Extract or simulate trade-by-trade P&L
    trade_results = extract_trade_results(results)
    
    # Extract or simulate daily returns
    daily_returns = extract_daily_returns(results)
    
    # Calculate enhanced metrics
    enhanced_metrics = calculate_enhanced_metrics(
        results,
        daily_returns=daily_returns,
        trade_results=trade_results
    )
    
    # Add enhanced metrics to the results dictionary with the 'enhanced_' prefix
    # to avoid name collisions with existing metrics
    for key, value in enhanced_metrics.items():
        results[f'enhanced_{key}'] = value
    
    return results

##=================================================[Command Line Interface]=================================================##

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Utility functions for trading systems")
    
    # Main commands (subparsers for different functions)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Cache terminal output command
    cache_parser = subparsers.add_parser("cache", help="Cache terminal output")
    cache_parser.add_argument("lines", type=int, nargs='?', default=100, 
                             help="Number of lines to capture")
    cache_parser.add_argument("-l", "--label", help="Label for this cache")
    
    # Show recent cache entries command
    show_parser = subparsers.add_parser("show", help="Show recent cache entries")
    show_parser.add_argument("count", type=int, nargs='?', default=5, 
                            help="Number of entries to show")
    
    # Read cache entry command
    read_parser = subparsers.add_parser("read", help="Read a cache entry")
    read_parser.add_argument("index", type=int, nargs="?", help="Index of the cache entry to read")
    
    # Market info command
    subparsers.add_parser("market", help="Show current market status")
    
    # Market time command
    subparsers.add_parser("markettime", help="Show time remaining in market session or until next session")
    
    # Sync trading data command
    subparsers.add_parser("sync", help="Sync trading data between backtester and live trader")
    
    # Trade info command
    trades_parser = subparsers.add_parser("trades", help="Show trade information")
    trades_parser.add_argument("--live", action="store_true", help="Show live trade data")
    
    return parser.parse_args()



def check_market_status():
    """Check and display current market status"""
    is_open, status = is_market_open()
    
    # Get the style based on status
    if is_open:
        status_style = "\033[38;2;0;200;0m"  # Green for open
    elif "not yet open" in status.lower():
        status_style = "\033[38;2;220;220;0m"  # Yellow for not yet open
    else:
        status_style = "\033[38;2;150;150;150m"  # Gray for closed
        
    print(f"Market Status: {status_style}{status}\033[0m")
    
    # Get additional date information
    today = datetime.now().date()
    last_trading_date = get_last_trading_date()
    next_trading_date = get_next_trading_day(today)
    
    print(f"Last Trading Day: \033[38;2;100;149;237m{last_trading_date}\033[0m")
    print(f"Next Trading Day: \033[38;2;100;149;237m{next_trading_date}\033[0m")
    
    return is_open
    
def check_market_time():
    """Check and display time remaining in the current market session or until next session"""
    nyse = mcal.get_calendar('NYSE')
    now = pd.Timestamp.now(tz='America/New_York')
    today_date = now.date()
    
    # Check if today is a trading day
    schedule = nyse.schedule(start_date=today_date, end_date=today_date)
    
    if schedule.empty:
        # Today is not a trading day, find the next trading day
        future_schedule = nyse.schedule(start_date=today_date + timedelta(days=1), 
                                        end_date=today_date + timedelta(days=10))
        if future_schedule.empty:
            print("\033[38;2;220;0;0mError: Unable to find next trading day\033[0m")
            return
            
        next_market_day = future_schedule.index[0].date()
        next_market_open = future_schedule.iloc[0]['market_open'].tz_convert('America/New_York')
        
        # Calculate time until market opens
        time_until_open = next_market_open - now
        days = time_until_open.days
        hours = time_until_open.seconds // 3600
        minutes = (time_until_open.seconds % 3600) // 60
        
        print(f"\033[38;2;150;150;150mMarket is currently closed\033[0m")
        print(f"Next market session: \033[38;2;100;149;237m{next_market_day.strftime('%A, %B %d')}\033[0m")
        print(f"Time until market opens: \033[38;2;100;149;237m{days} days, {hours} hours, {minutes} minutes\033[0m")
        
    else:
        # Today is a trading day
        market_open = schedule.iloc[0]['market_open'].tz_convert('America/New_York')
        market_close = schedule.iloc[0]['market_close'].tz_convert('America/New_York')
        
        if now < market_open:
            # Market not yet open
            time_until_open = market_open - now
            hours = time_until_open.seconds // 3600
            minutes = (time_until_open.seconds % 3600) // 60
            
            print(f"\033[38;2;220;220;0mMarket will open today at {market_open.strftime('%H:%M:%S')}\033[0m")
            print(f"Time until market opens: \033[38;2;220;220;0m{hours} hours, {minutes} minutes\033[0m")
            
        elif now > market_close:
            # Market already closed
            future_schedule = nyse.schedule(start_date=today_date + timedelta(days=1), 
                                          end_date=today_date + timedelta(days=10))
            if future_schedule.empty:
                print("\033[38;2;220;0;0mError: Unable to find next trading day\033[0m")
                return
                
            next_market_day = future_schedule.index[0].date()
            next_market_open = future_schedule.iloc[0]['market_open'].tz_convert('America/New_York')
            
            # Calculate time until market opens
            time_until_open = next_market_open - now
            days = time_until_open.days
            hours = time_until_open.seconds // 3600
            minutes = (time_until_open.seconds % 3600) // 60
            
            print(f"\033[38;2;150;150;150mMarket closed at {market_close.strftime('%H:%M:%S')}\033[0m")
            print(f"Next market session: \033[38;2;100;149;237m{next_market_day.strftime('%A, %B %d')}\033[0m")
            print(f"Time until market opens: \033[38;2;100;149;237m{days} days, {hours} hours, {minutes} minutes\033[0m")
            
        else:
            # Market is open
            time_remaining = market_close - now
            hours = time_remaining.seconds // 3600
            minutes = (time_remaining.seconds % 3600) // 60
            
            print(f"\033[38;2;0;200;0mMarket is open\033[0m")
            print(f"Market closes at: \033[38;2;0;200;0m{market_close.strftime('%H:%M:%S')}\033[0m")
            print(f"Time remaining in session: \033[38;2;0;200;0m{hours} hours, {minutes} minutes\033[0m")

def display_trades(is_live=False):
    """Display current trades and signals"""
    try:
        df = read_trading_data(is_live=is_live)
        
        if df.empty:
            print("\033[38;2;220;220;0mNo trading data found\033[0m")
            return
        
        # Find open positions
        open_positions = df[df['IsCurrentlyBought'] == True]
        
        # Find buy signals
        buy_signals = df[
            (df['IsCurrentlyBought'] == False) & 
            (df['LastBuySignalDate'].notna())
        ]
        
        # Get last trading date
        last_date = get_last_trading_date()
        prev_day = get_previous_trading_day(last_date)
        
        # Filter for recent signals
        recent_signals = buy_signals[buy_signals['LastBuySignalDate'].dt.date >= prev_day] if not buy_signals.empty else buy_signals
        
        # Display open positions
        if not open_positions.empty:
            print("\n\033[1mOpen Positions:\033[0m")
            print(f"{'Symbol':<8} {'Entry Date':<12} {'Entry Price':<12} {'Position Size':<14}")
            print("-" * 50)
            
            for _, pos in open_positions.iterrows():
                entry_date = pos['LastBuySignalDate'].strftime('%Y-%m-%d') if pd.notna(pos['LastBuySignalDate']) else 'N/A'
                entry_price = f"${pos['LastBuySignalPrice']:.2f}" if pd.notna(pos['LastBuySignalPrice']) else 'N/A'
                pos_size = f"${pos['PositionSize']:.2f}" if pd.notna(pos['PositionSize']) else 'N/A'
                
                print(f"{pos['Symbol']:<8} {entry_date:<12} {entry_price:<12} {pos_size:<14}")
        else:
            print("\n\033[38;2;150;150;150mNo open positions\033[0m")
        
        # Display recent buy signals
        if not recent_signals.empty:
            print("\n\033[1mRecent Buy Signals:\033[0m")
            print(f"{'Symbol':<8} {'Signal Date':<12} {'Signal Price':<12} {'Probability':<14}")
            print("-" * 50)
            
            for _, signal in recent_signals.iterrows():
                signal_date = signal['LastBuySignalDate'].strftime('%Y-%m-%d') if pd.notna(signal['LastBuySignalDate']) else 'N/A'
                signal_price = f"${signal['LastBuySignalPrice']:.2f}" if pd.notna(signal['LastBuySignalPrice']) else 'N/A'
                probability = f"{signal['UpProbability']*100:.1f}%" if pd.notna(signal['UpProbability']) else 'N/A'
                
                print(f"{signal['Symbol']:<8} {signal_date:<12} {signal_price:<12} {probability:<14}")
        else:
            print("\n\033[38;2;150;150;150mNo recent buy signals\033[0m")
            
    except Exception as e:
        print(f"\033[38;2;220;0;0mError displaying trades: {str(e)}\033[0m")


def organize_trading_data(max_signals=5):
    """
    Organize trading data into the production directory structure.
    
    Args:
        max_signals: Maximum number of signals to include for tomorrow
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logging.info("Organizing trading data for production...")
        
        # Define file paths
        source_files = {
            'buy_signals': '_Buy_Signals.parquet',
            'live_trades': '_Live_trades.parquet',
            'trade_history': 'trade_history.parquet'
        }
        
        dest_files = {
            'signals': 'Data/Production/BacktestData/signals.parquet',
            'backtest_results': 'Data/Production/BacktestData/backtest_results.parquet',
            'active_positions': 'Data/Production/LiveTradingData/active_positions.parquet',
            'completed_trades': 'Data/Production/LiveTradingData/completed_trades.parquet',
            'pending_signals': 'Data/Production/LiveTradingData/pending_signals.parquet'
        }
        
        # Ensure directories exist
        for file_path in dest_files.values():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Calculate tomorrow's date (next trading day)
        today = datetime.now().date()
        tomorrow = get_next_trading_day(today)
        logging.info(f"Today: {today}, Next Trading Day: {tomorrow}")
        
        # 1. Extract and process top signals for tomorrow
        if os.path.exists(source_files['buy_signals']):
            signals_df = pd.read_parquet(source_files['buy_signals'])
            
            # Select top signals based on UpProbability
            top_signals = signals_df[
                (signals_df['IsCurrentlyBought'] == False) &
                (signals_df['UpProbability'] > 0.65)  # Minimum probability threshold
            ].sort_values('UpProbability', ascending=False).head(max_signals)
            
            if not top_signals.empty:
                # Update signal date to tomorrow
                top_signals = top_signals.copy()  # Avoid SettingWithCopyWarning
                top_signals['LastBuySignalDate'] = pd.Timestamp(tomorrow)
                top_signals['SignalExpiration'] = pd.Timestamp(tomorrow + timedelta(days=1))
                top_signals['TargetDate'] = pd.Timestamp(tomorrow)
                
                # Save to production signals file
                top_signals.to_parquet(dest_files['signals'], index=False)
                logging.info(f"Saved {len(top_signals)} top signals for {tomorrow} to {dest_files['signals']}")
                
                # Also update pending signals
                top_signals.to_parquet(dest_files['pending_signals'], index=False)
                logging.info(f"Updated pending signals for tomorrow")
            else:
                logging.warning("No valid signals found for tomorrow")
        else:
            logging.warning(f"Buy signals file {source_files['buy_signals']} not found")
        
        # 2. Update active positions
        active_positions = None
        
        # First check Buy_Signals for active positions
        if os.path.exists(source_files['buy_signals']):
            buy_signals_df = pd.read_parquet(source_files['buy_signals'])
            active_from_backtester = buy_signals_df[buy_signals_df['IsCurrentlyBought'] == True]
            
            if not active_from_backtester.empty:
                active_positions = active_from_backtester
                logging.info(f"Found {len(active_positions)} active positions in {source_files['buy_signals']}")
        
        # Then check Live_trades for more active positions
        if os.path.exists(source_files['live_trades']):
            live_trades_df = pd.read_parquet(source_files['live_trades'])
            active_from_live = live_trades_df[live_trades_df['IsCurrentlyBought'] == True]
            
            if not active_from_live.empty:
                if active_positions is None:
                    active_positions = active_from_live
                else:
                    # Merge positions, keeping most recent entries
                    combined = pd.concat([active_positions, active_from_live])
                    # Remove duplicates, keeping most recent by LastBuySignalDate
                    combined = combined.sort_values('LastBuySignalDate', ascending=False)
                    active_positions = combined.drop_duplicates('Symbol', keep='first')
                
                logging.info(f"Combined {len(active_positions)} total active positions")
        
        if active_positions is not None and not active_positions.empty:
            # Add last update timestamp
            active_positions = active_positions.copy()
            active_positions['LastUpdate'] = pd.Timestamp.now()
            
            # Save to production active positions file
            active_positions.to_parquet(dest_files['active_positions'], index=False)
            logging.info(f"Saved {len(active_positions)} active positions to {dest_files['active_positions']}")
        else:
            logging.info("No active positions found")
        
        # 3. Update backtest results with trade history
        if os.path.exists(source_files['trade_history']):
            trade_history_df = pd.read_parquet(source_files['trade_history'])
            
            # Save to production backtest results file
            trade_history_df.to_parquet(dest_files['backtest_results'], index=False)
            logging.info(f"Saved {len(trade_history_df)} trade history records to {dest_files['backtest_results']}")
            
            # Update completed trades as well
            completed_trades = trade_history_df.copy()
            # Add any additional metadata for completed trades
            completed_trades['RecordType'] = 'Backtest'
            
            # Save to production completed trades file
            completed_trades.to_parquet(dest_files['completed_trades'], index=False)
            logging.info(f"Updated completed trades with {len(completed_trades)} records")
        else:
            logging.warning(f"Trade history file {source_files['trade_history']} not found")
        
        logging.info("Trading data organization completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error organizing trading data: {str(e)}")
        logging.error(traceback.format_exc())
        return False




def verify_signal_handoff(max_signals=5):
    """
    Verify the entire signal handoff process from backtester to live trader.
    
    This function:
    1. Verifies signals are generated in the backtester
    2. Checks the synchronization process
    3. Verifies the organization of data for the live trader
    4. Tests that the live trader can read the signals
    
    Args:
        max_signals: Maximum number of signals to process
        
    Returns:
        True if the process is working correctly, False otherwise
    """
    import os
    import pandas as pd
    import logging
    from datetime import datetime, timedelta
    
    # Set up logging
    logger = logging.getLogger("verify_signal_handoff")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info("Starting verification of signal handoff from backtester to live trader")
    
    # Define file paths
    files = {
        'backtest': '_Buy_Signals.parquet',
        'live': '_Live_trades.parquet',
        'prod_signals': 'Data/Production/BacktestData/signals.parquet',
        'pending_signals': 'Data/Production/LiveTradingData/pending_signals.parquet',
    }
    
    # Step 1: Verify backtester signals exist
    if not os.path.exists(files['backtest']):
        logger.error(f"Backtest signals file not found: {files['backtest']}")
        return False
    
    try:
        backtest_df = pd.read_parquet(files['backtest'])
        if backtest_df.empty:
            logger.warning("Backtest signals file is empty")
        else:
            logger.info(f"Backtester signals file has {len(backtest_df)} records")
    except Exception as e:
        logger.error(f"Error reading backtest signals: {str(e)}")
        return False
    
    # Step 2: Verify synchronization with live trader
    try:
        from Util import sync_trading_data
        
        # Backup existing live trades file if it exists
        live_backup = None
        if os.path.exists(files['live']):
            live_backup = pd.read_parquet(files['live'])
            logger.info(f"Backed up existing live trades file with {len(live_backup)} records")
        
        # Run synchronization
        logger.info("Running trading data synchronization...")
        sync_result = sync_trading_data()
        
        # Verify live trades file exists after sync
        if not os.path.exists(files['live']):
            logger.error("Live trades file not created by sync_trading_data")
            # Restore backup if we had one
            if live_backup is not None:
                live_backup.to_parquet(files['live'], index=False)
            return False
        
        live_df = pd.read_parquet(files['live'])
        logger.info(f"Live trades file has {len(live_df)} records after sync")
        
        # Restore backup if we had one
        if live_backup is not None:
            live_backup.to_parquet(files['live'], index=False)
            logger.info("Restored original live trades file")
        
    except Exception as e:
        logger.error(f"Error during synchronization: {str(e)}")
        return False
    
    # Step 3: Verify organization of data for tomorrow
    try:
        from Util import organize_trading_data, get_next_trading_day
        
        # Run organization
        logger.info("Running trading data organization...")
        org_result = organize_trading_data(max_signals=max_signals)
        
        if not org_result:
            logger.error("organize_trading_data reported failure")
            return False
        
        # Verify production files exist
        for key in ['prod_signals', 'pending_signals']:
            if not os.path.exists(files[key]):
                logger.error(f"Production file not created: {files[key]}")
                return False
            
            try:
                df = pd.read_parquet(files[key])
                logger.info(f"{key} file has {len(df)} records")
                
                # Check if signals are for tomorrow
                if 'TargetDate' in df.columns and not df.empty:
                    tomorrow = get_next_trading_day(datetime.now().date())
                    signals_for_tomorrow = df[df['TargetDate'].dt.date == tomorrow]
                    logger.info(f"Found {len(signals_for_tomorrow)} signals for tomorrow ({tomorrow})")
                
            except Exception as e:
                logger.error(f"Error reading {key}: {str(e)}")
                return False
    
    except Exception as e:
        logger.error(f"Error during organization: {str(e)}")
        return False
    
    # Step 4: Verify live trader can read the signals
    try:
        from Util import get_buy_signals
        
        logger.info("Testing if live trader can read signals...")
        signals = get_buy_signals(is_live=True)
        
        if not signals:
            logger.warning("No signals returned by get_buy_signals")
        else:
            logger.info(f"get_buy_signals returned {len(signals)} signals")
            for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                logger.info(f"Signal {i+1}: {signal.get('Symbol')} - " 
                           f"UpProb: {signal.get('UpProbability', 0):.4f}, "
                           f"Price: {signal.get('LastBuySignalPrice', 0):.2f}")
    
    except Exception as e:
        logger.error(f"Error testing live trader signal reading: {str(e)}")
        return False
    
    # If we got here, everything works!
    logger.info("✅ Signal handoff verification completed successfully!")
    logger.info("The backtester is properly generating signals for the live trader.")
    return True




def run_post_backtest_organization():
    """Run post-backtest organization to prepare for tomorrow's trading"""
    try:
        logger = get_logger(__name__ if __name__ != "__main__" else "post_backtest_organization")
        logger.info("Running post-backtest organization")
        
        # Import the function directly from the module
        # This is already defined in Util.py so no need to redefine it here
        # Just make sure to import it if not already imported
        from Util import organize_trading_data
        
        # Organize data with 5 top signals
        success = organize_trading_data(max_signals=5)
        if success:
            logger.info("Successfully organized trading data for tomorrow")
        else:
            logger.error("Failed to organize trading data")
        
        return success
    except Exception as e:
        logger = logging.getLogger(__name__ if __name__ != "__main__" else "post_backtest_organization")
        logger.error(f"Error in post-backtest organization: {str(e)}")
        logger.error(traceback.format_exc())
        return False








def main():
    """Main command line interface function"""
    args = parse_args()
    
    # If no command provided, show help
    if not args.command:
        parse_args.__globals__['parser'].print_help()
        return
    
    # Handle different commands
    if args.command == "cache":
        cache_terminal(args.lines, args.label)
    
    elif args.command == "show":
        show_cache_entries(args.count)
    
    elif args.command == "read":
        read_cache(index=args.index)
    
    elif args.command == "market":
        check_market_status()
        
    elif args.command == "markettime":
        check_market_time()
    
    elif args.command == "sync":
        success = sync_trading_data()
        if success:
            print("\033[38;2;0;200;0mTrading data synchronized successfully\033[0m")
        else:
            print("\033[38;2;220;0;0mError synchronizing trading data\033[0m")
    
    elif args.command == "trades":
        display_trades(is_live=args.live)
    
    else:
        print(f"\033[38;2;220;0;0mUnknown command: {args.command}\033[0m")

if __name__ == "__main__":
    main()