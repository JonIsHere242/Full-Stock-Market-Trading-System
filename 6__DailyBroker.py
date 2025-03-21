#!/usr/bin/env python
import random
import sys
import time
import uuid
import os
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import backtrader as bt
from backtrader_ib_insync import IBStore
import ib_insync as ibi
import pandas as pd
import numpy as np
import exchange_calendars as ec
import logging
from Util import * 




# Debug mode toggle
DEBUG_MODE = True

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PAPER_PORT = 7497
DEFAULT_LIVE_PORT = 7497
MAX_RECONNECT_ATTEMPTS = 3
CONNECTION_TIMEOUT = 20.0
SIGNALS_FILE = 'Data/Production/LiveTradingData/pending_signals.parquet'
POSITIONS_FILE = 'Data/Production/LiveTradingData/active_positions.parquet'
COMPLETED_TRADES_FILE = 'Data/Production/LiveTradingData/completed_trades.parquet'



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
            logger.error(f"Error reading signals file: {str(e)}")
    
    return False



class Colors:
    RESET = "\033[0m"
    INFO = "\033[38;2;0;200;0m"      # Green
    WARN = "\033[38;2;220;220;0m"    # Yellow
    ERROR = "\033[38;2;220;0;0m"     # Red
    DETAIL = "\033[38;2;100;149;237m" # Cornflower blue
    DEBUG = "\033[38;2;0;180;180m"   # Cyan
    SUCCESS = "\033[38;2;50;220;50m" # Bright Green
    TRACE = "\033[38;2;180;180;180m" # Gray
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    TIMESTAMP = "\033[38;2;150;150;150m" # Gray for timestamps






def dprint(message, level="INFO", show_timestamp=True, indent=0, logger=None):
    """Enhanced debug print function with colorized output by level.
    
    Args:
        message: Message to print
        level: One of "INFO", "WARN", "ERROR", "DETAIL", "DEBUG", "SUCCESS", "TRACE"
        show_timestamp: Whether to include timestamp
        indent: Indentation level (number of spaces)
        logger: Optional logger to also log to file
    """
    if not DEBUG_MODE and level == "TRACE":
        return  # Skip trace messages when not in debug mode
        
    color = getattr(Colors, level, Colors.INFO)
    level_str = f"[{level}]".ljust(8)
    
    timestamp = ""
    if show_timestamp:
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        timestamp = f"{Colors.TIMESTAMP}{current_time}{Colors.RESET} "
        
    indent_str = " " * indent
    
    # Print to console with color
    print(f"{timestamp}{indent_str}{color}{level_str}{Colors.RESET} {message}")
    
    # Also log to file via logger if provided
    if logger:
        log_method = getattr(logger, level.lower(), logger.info)
        if level.lower() not in ['detail', 'trace', 'success']:  # These aren't standard logger levels
            log_method(f"{indent_str}{message}")
        else:
            logger.info(f"[{level}] {indent_str}{message}")



















def get_open_positions(ib):
    """Get current open positions from IB."""
    try:
        positions = ib.positions()
        positions_list = []
        for position in positions:
            if position.position != 0:
                contract = position.contract
                positions_list.append(contract.symbol)
                logger.info(f'Position found: {contract.symbol}, {position.position}')
        
        if not positions_list:
            logger.info('No open positions found')
        
        return positions_list
    except Exception as e:
        logger.error(f'Error fetching positions: {e}')
        return []

nyse = ec.get_calendar('XNYS')

def wait_for_market_open(tz=ZoneInfo('America/New_York'), max_wait_minutes=180, manual_override=False):
    """
    Blocks execution until the NYSE is open or until max_wait_minutes is reached.
    Returns True if the market opened within that time, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        logger.info("Manual override enabled - bypassing market open check")
        return True
        
    logger.info("Checking if the market is currently open...")

    # Get the current time in NYSE timezone
    now = pd.Timestamp.now(tz)

    # Extract the date as a timezone-naive date object
    current_date = now.date()

    # Check if today is a trading day
    if nyse.is_session(current_date):
        try:
            # Get market open and close times
            market_open = nyse.session_open(current_date)
            market_close = nyse.session_close(current_date)
        except Exception as e:
            logger.error(f"Error fetching session times for {current_date}: {e}")
            return False

        # Check if the current time is within trading hours
        if market_open <= now <= market_close:
            logger.info("Market is open.")
            return True

    # If the market is not open, determine the next open time
    next_open = nyse.next_open(now)
    if not next_open:
        logger.warning("No upcoming market open found. Possibly a holiday.")
        return False

    logger.info(f"Waiting for the market to open at {next_open}...")

    # Calculate maximum wait time in seconds
    max_wait_seconds = max_wait_minutes * 60
    wait_seconds = 0
    interval = 30  # Check every 30 seconds

    # Wait loop
    while wait_seconds < max_wait_seconds:
        now = pd.Timestamp.now(tz)
        if now >= next_open:
            logger.info("Market has opened.")
            return True
        time.sleep(interval)
        wait_seconds += interval

    logger.warning("Market did not open within the expected wait time.")
    return False

def is_nyse_open(manual_override=False):
    """
    Returns True if the NYSE is currently open, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        logger.info("Manual override enabled - reporting market as open")
        return True
        
    tz_nyse = ZoneInfo('America/New_York')
    now_nyse = pd.Timestamp.now(tz_nyse)
    current_date = now_nyse.date()

    # Check if today is a trading day
    if not nyse.is_session(current_date):
        logger.info("Today is not a trading day.")
        return False

    # Get market open and close times for the current date
    try:
        market_open = nyse.session_open(current_date)
        market_close = nyse.session_close(current_date)
    except Exception as e:
        logger.error(f"Error fetching session times for {current_date}: {e}")
        return False

    # Check if current time is within trading hours
    if market_open <= now_nyse <= market_close:
        logger.info("NYSE is currently open.")
        return True
    else:
        logger.info("NYSE is currently closed.")
        return False

class FixedCommissionScheme(bt.CommInfoBase):
    """Fixed commission scheme for IB."""
    params = (
        ('commission', 3.0),  # Fixed commission per trade
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission  # Return fixed commission

class PositionSizer:
    """Position sizing logic identical to backtester."""
    def __init__(self, risk_per_trade=1.0, max_position_pct=0.20, min_position_pct=0.05, reserve_pct=0.10):
        self.risk_per_trade = risk_per_trade  # Risk per trade in percent of account
        self.max_position_pct = max_position_pct  # Maximum position size as % of account
        self.min_position_pct = min_position_pct  # Minimum position size as % of account
        self.reserve_pct = reserve_pct  # Cash reserve percentage
    
    def calculate_position_size(self, account_value, cash, price, atr, max_positions):
        workable_capital = account_value * (1.0 - self.reserve_pct) # Check if we have enough free cash
        
        if cash < account_value * self.reserve_pct: # Don't take a position if cash is too low
            return 0
            
        risk_amount = account_value * (self.risk_per_trade / 100.0) # Calculate position size based on risk and ATR
        risk_per_share = 2.0 * atr # Use 2x ATR as the risk per share
        
        if risk_per_share <= 0.01:  # Avoid division by zero or very small numbers
            risk_per_share = 0.01
            
        shares_by_risk = risk_amount / risk_per_share
        
        min_position = (account_value * self.min_position_pct) / price # Calculate position limits based on account percentages
        max_position = (account_value * self.max_position_pct) / price
        
        # Position size is the middle ground between risk-based and percentage-based
        position_size = np.clip(shares_by_risk, min_position, max_position)
        
        # Also consider max positions to avoid over-concentration
        position_size = min(position_size, workable_capital / (price * max_positions))
        position_size = int(position_size)
        
        # Final check: Ensure we have enough cash
        if position_size * price > cash * 0.95:  # Leave some buffer
            position_size = int(cash * 0.95 / price)
            
        return max(0, position_size)











class StockSniperLive(bt.Strategy):
    """Live trading strategy matching the backtest StockSniper."""
    
    # Set parameters directly from the shared constants
    params = STRATEGY_PARAMS

    def __init__(self):
        """Initialize the strategy."""
        logger.info("Initializing StockSniperLive strategy...")

        self.barCounter = 0  # ADDED: Initialize barCounter
        self.start_time = datetime.now()  # Track strategy start time

        self.position_sizer = PositionSizer(
            risk_per_trade=self.p.risk_per_trade_pct,
            max_position_pct=self.p.max_position_pct,
            min_position_pct=self.p.min_position_pct,
            reserve_pct=self.p.reserve_percent
        )
        
        # Order tracking
        self.order_dict = {}
        self.entry_prices = {}
        self.position_dates = {}
        self.trailing_stops = {}
        
        # Data management
        self.data_ready = {d: False for d in self.datas}
        self.position_closed = {d._name: False for d in self.datas}
        self.last_sync_time = datetime.now()
        
        # Initialize live trading data
        self.trading_data = read_trading_data(is_live=True)
        
        # ATR indicators for all data feeds
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=14)
            
        # Add a timer for heartbeat and sync
        self.add_timer(
            when=bt.Timer.SESSION_START,
            offset=timedelta(seconds=0),
            repeat=timedelta(minutes=5),  # Sync every 5 minutes
            weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        """Enhanced timer notification with data verification"""
        elapsed_mins = (datetime.now() - self.start_time).total_seconds() / 60
        current_time = datetime.now().strftime("%H:%M:%S")

        # Periodic data verification
        if self.barCounter > 0 and self.barCounter % 10 == 0:  # Every 10 bars
            self.verify_data_status()

        # Sync with backtester signals
        try:
            sync_trading_data()
            self.trading_data = read_trading_data(is_live=True)
            logger.info("Successfully synchronized with backtester signals")
        except Exception as e:
            logger.error(f"Error synchronizing with backtester: {e}")

        # Synchronize positions
        self.synchronize_positions()

        # Log portfolio value
        portfolio_value = self.broker.getvalue()
        logger.info(f"Heartbeat: {current_time} - Running for {elapsed_mins:.1f} minutes")
        logger.info(f"Current portfolio value: ${portfolio_value:.2f}")

        # Check if we should exit
        if elapsed_mins >= 5:  # 5 minute runtime (adjust as needed)
            logger.info('Strategy runtime threshold reached. Starting shutdown sequence...')
            self.env.runstop()
            try:
                store = self.broker.store
                if hasattr(store, 'disconnect'):
                    store.disconnect()
                    logger.info('Disconnected from Interactive Brokers')
            except Exception as e:
                logger.error(f"Error disconnecting from IB: {e}")

            logger.info('Shutdown sequence complete. Exiting in 5 seconds...')
            time.sleep(5)
            sys.exit(0)







    def sync_with_backtester(self):
        """Synchronize with signals from the backtester."""
        try:
            # Use the shared synchronization function
            sync_trading_data()
            
            # Reload the data
            self.trading_data = read_trading_data(is_live=True)
            logger.info("Synchronized with backtester signals.")
            
        except Exception as e:
            logger.error(f"Error synchronizing with backtester: {e}")
            logger.error(traceback.format_exc())





    def notify_data(self, data, status, *args, **kwargs):
        """Enhanced data status notification handler with better logging."""
        status_name = data._getstatusname(status)
        print(f"Data status change for {data._name}: {status_name}")

        if status == data.LIVE:
            self.data_ready[data] = True
            logger.info(f"{data._name} is now LIVE and ready for trading")
        elif status == data.DELAYED:
            logger.info(f"{data._name} is in DELAYED mode - data is historical/backfilling")
        elif status == data.DISCONNECTED:
            self.data_ready[data] = False
            logger.warning(f"{data._name} DISCONNECTED - will not process until reconnected")
        elif status == data.CONNBROKEN:
            self.data_ready[data] = False
            logger.warning(f"{data._name} connection BROKEN - attempting to reconnect")
        elif status == data.NOTSUBSCRIBED:
            self.data_ready[data] = False
            logger.error(f"{data._name} NOT SUBSCRIBED - check permissions")
        elif status == data.CONNECTED:
            logger.info(f"{data._name} CONNECTED - waiting for data to go LIVE")
        else:
            logger.info(f"{data._name} status changed to {status_name}")

        # Log the full state of data_ready after each notification
        ready_feeds = [d._name for d, is_ready in self.data_ready.items() if is_ready]
        not_ready_feeds = [d._name for d, is_ready in self.data_ready.items() if not is_ready]
        logger.info(f"Data ready status: {len(ready_feeds)}/{len(self.data_ready)} feeds ready")
        if not_ready_feeds:
            logger.info(f"Waiting for feeds: {', '.join(not_ready_feeds)}")






    def notify_order(self, order):
        """Order notification handler matching backtester's logic."""
        if order.status in [order.Submitted, order.Accepted]:
            self.order_dict[order.ref] = {
                'order': order,
                'data_name': order.data._name,
                'status': order.getstatusname()
            }
            logger.info(f"Order for {order.data._name} {order.getstatusname()}")
            return

        # Handle completed orders
        if order.status in [order.Completed]:
            symbol = order.data._name

            if order.isbuy():
                # Process buy order
                self.handle_buy_order(order)

                # Check if this was a main order in a bracket that needs protection
                if hasattr(order, 'parent') and order.parent is None:  # It's a parent order
                    # Find any child orders
                    child_orders = []
                    for child_ref, child_info in self.order_dict.items():
                        child_order = child_info.get('order')
                        if (child_order and hasattr(child_order, 'parent') and 
                            child_order.parent is not None and child_order.parent.ref == order.ref):
                            child_orders.append(child_order)

                    # If we expected children but don't have any or they're not active, place protection
                    if len(child_orders) < 2:  # Should have at least stop and limit children
                        logger.warning(f"Bracket order children missing for {symbol}. Creating protection orders.")
                        self.handle_bracket_failure(symbol, order.executed.size, order.executed.price)
                    else:
                        # Check if any children were cancelled or rejected
                        for child in child_orders:
                            if child.status in [order.Canceled, order.Rejected]:
                                logger.warning(f"Child order failed for {symbol}. Creating protection orders.")
                                self.handle_bracket_failure(symbol, order.executed.size, order.executed.price)
                                break
                            
            elif order.issell():
                self.handle_sell_order(order)
                self.position_closed[order.data._name] = True

            logger.info(f"Order for {symbol} {order.getstatusname()}")

        # Handle failed orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            symbol = order.data._name

            # Special handling for child order failures in a bracket
            if hasattr(order, 'parent') and order.parent is not None:
                parent_ref = order.parent.ref
                if parent_ref in self.order_dict:
                    parent_info = self.order_dict[parent_ref]
                    parent_status = parent_info.get('order').status

                    # If parent already completed but child failed, we need protection orders
                    if parent_status == order.Completed:
                        logger.warning(f"Child order {order.ref} failed for {symbol} with completed parent. Creating protection.")
                        parent_order = parent_info.get('order')
                        self.handle_bracket_failure(symbol, parent_order.executed.size, parent_order.executed.price)

            logger.warning(f'Order {symbol} failed with status: {order.getstatusname()}')

        # Remove completed or failed orders from dictionary
        if order.ref in self.order_dict:
            # Keep track of bracket parent-child relationships before removing
            if order.status == order.Completed and order.isbuy():
                # For completed buy orders, keep the entry in the dictionary temporarily
                # but mark it as completed for the protection order logic
                self.order_dict[order.ref]['status'] = 'Completed'
            else:
                del self.order_dict[order.ref]
                



    def handle_buy_order(self, order):
        """Handle completed buy orders with proper trade data updating."""
        symbol = order.data._name
        entry_price = order.executed.price
        entry_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logger.info(f"Buy order for {symbol} completed. Size: {position_size}, Price: {entry_price}")

        try:
            # First read existing data
            df = read_trading_data(is_live=True)

            # Create new trade data
            new_data = {
                'Symbol': symbol,
                'LastBuySignalDate': pd.Timestamp(entry_date),
                'LastBuySignalPrice': entry_price,
                'IsCurrentlyBought': True,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.Timestamp(entry_date),
                'UpProbability': 0.0,  # You might want to pass this as a parameter
                'PositionSize': position_size
            }

            # Update or append the data
            if symbol in df['Symbol'].values:
                for col, value in new_data.items():
                    df.loc[df['Symbol'] == symbol, col] = value
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            # Write back to parquet
            write_trading_data(df, is_live=True)
            logger.info(f"Successfully updated trade data for buy order: {symbol}")

            # Store entry information
            self.entry_prices[order.data] = entry_price
            self.position_dates[order.data] = entry_date

        except Exception as e:
            logger.error(f"Error updating trade data for {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    def handle_sell_order(self, order):
        """Handle completed sell orders with proper trade data updating."""
        symbol = order.data._name
        exit_price = order.executed.price
        exit_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logger.info(f"Sell order for {symbol} completed. Size: {position_size}, Price: {exit_price}")

        # Update the Parquet file
        update_trade_data(symbol, 'sell', exit_price, exit_date, 0, is_live=True)

        # Clean up tracking data
        data = order.data
        if data in self.entry_prices:
            del self.entry_prices[data]
        if data in self.position_dates:
            del self.position_dates[data]
        if data in self.trailing_stops:
            del self.trailing_stops[data]
            
        logger.info(f"Cleaned up tracking data for {symbol} after sell completion")
        
    def handle_bracket_failure(self, symbol, size, avg_price):
        """
        Handle cases where a buy order was filled but the bracket orders failed.
        This places standalone stop-loss and take-profit orders.
        """
        try:
            # Find the data feed for this symbol
            data = None
            for d in self.datas:
                if d._name == symbol:
                    data = d
                    break
                
            if data is None:
                logger.error(f"Cannot create protection orders for {symbol}: data feed not found")
                return

            # Calculate stop and take-profit prices
            stop_price = avg_price * (1 - self.p.stop_loss_percent/100)
            take_profit_price = avg_price * (1 + self.p.take_profit_percent/100)

            # Round prices properly
            tick_size = 0.01 if avg_price >= 1.0 else 0.0001
            stop_price = round(stop_price / tick_size) * tick_size
            take_profit_price = round(take_profit_price / tick_size) * tick_size

            # Create OCO (One-Cancels-Other) group ID
            oco_id = str(uuid.uuid4())

            # Place trailing stop
            trailing_stop = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.StopTrail,
                trailpercent=self.p.trailing_stop_atr_multiple / 100,
                transmit=True,
                oco=oco_id
            )

            # Place take-profit limit
            take_profit = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.Limit,
                price=take_profit_price,
                transmit=True,
                oco=oco_id
            )

            logger.info(f"Created standalone protection orders for {symbol}: "
                       f"Trailing stop at {self.p.trailing_stop_atr_multiple}%, "
                       f"Take-profit at {take_profit_price}")

            return trailing_stop, take_profit

        except Exception as e:
            logger.error(f"Error creating protection orders for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return None, None
            





    def prenext(self):
        """
        Pre-next is called when we don't have enough bars for all data feeds.
        We'll use it to track which symbols have data and which are still waiting.
        """
        self.barCounter += 1

        # Log the basic information
        logger.info(f"Prenext bar {self.barCounter} - waiting for more data")

        # Count active feeds and available data points
        ready_feeds = []
        waiting_feeds = []

        for data in self.datas:
            # Check if this is a "raw" feed or "resampled" feed based on naming convention
            if "_5sec" in data._name:
                logger.info(f"RAW {data._name} has {len(data)} bars")
            else:
                logger.info(f"RESAMPLED {data._name} has {len(data)} bars")

            is_raw_feed = "_5sec" in data._name if hasattr(data, '_name') else False

            if is_raw_feed:
                continue

            # Only report on the resampled feeds (those without "_5sec" in name)
            symbol = data._name

            if len(data) > 0:
                # Log the number of bars for this feed
                ohlcv_str = (
                f"Data feed {symbol} ({len(data)} bars) - "
                f"O:{data.open[0]:.2f}, H:{data.high[0]:.2f}, "
                f"L:{data.low[0]:.2f}, C:{data.close[0]:.2f}, "
                f"V:{int(data.volume[0])}"
                )
                dprint(ohlcv_str, level="INFO")
                ready_feeds.append(symbol)
                
            else:
                dprint(f"Data feed {symbol} has no bars yet", level="WARN")
                waiting_feeds.append(symbol)

        # Log overall data feed status
        if self.barCounter % 10 == 0:  # Only log this every 10 bars to reduce spam
            logger.info(f"Data feed status: {len(ready_feeds)}/{len(ready_feeds) + len(waiting_feeds)} feeds have data")
            if waiting_feeds:
                logger.info(f"Still waiting for data: {', '.join(waiting_feeds)}")

            # Every 20 bars, check if we've been waiting too long
            if self.barCounter >= 20 and waiting_feeds:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                logger.warning(f"Waited {elapsed_time:.1f} seconds but still missing data for: {', '.join(waiting_feeds)}")




    def next(self):
        """
        Main strategy logic executed on each bar, with improved data feed handling.
        We only process feeds that have the proper name format (no "_5sec" suffix).
        """
        # Increment bar counter
        self.barCounter += 1

        # Get current date safely
        try:
            current_date = self.datetime.date(0)
        except Exception:
            current_date = datetime.now().date()

        logger.info(f"Processing bar #{self.barCounter} at {datetime.now()} for date {current_date}")

        # Get accurate position data from IB
        active_positions = self.synchronize_positions()

        # Refresh trading data
        self.trading_data = read_trading_data(is_live=True)

        # Log current state
        logger.info(f"Active positions from IB: {list(active_positions.keys())}")
        logger.info(f"Portfolio value: ${self.broker.getvalue():.2f}")

        # Check market status
        market_open = is_nyse_open()
        if not market_open:
            logger.info(f"Market is closed. Skipping trading logic.")
            return

        # Process only the resampled data feeds (not the raw 5-second feeds)
        for data in self.datas:
            # Skip raw feeds (those with "_5sec" in the name)
            is_raw_feed = "_5sec" in data._name if hasattr(data, '_name') else False
            if is_raw_feed:
                continue

            # Skip feeds that aren't marked as ready
            if not self.data_ready.get(data, False):
                continue

            symbol = data._name

            # Log current bar data
            if len(data) > 0:
                logger.info(f"{symbol} Bar: O: {data.open[0]:.2f}, H: {data.high[0]:.2f}, L: {data.low[0]:.2f}, C: {data.close[0]:.2f}, V: {int(data.volume[0])}")

            # Use active positions from IB
            position_size = active_positions.get(symbol, {}).get('size', 0)

            # Process existing positions for potential exits
            if position_size > 0 and not self.position_closed[symbol]:
                avg_cost = active_positions.get(symbol, {}).get('avg_cost', 0)
                logger.info(f"{symbol} - Current position: {position_size} shares at avg price: ${avg_cost:.2f}")

                # Evaluate whether to sell
                self.evaluate_sell_conditions(data, current_date, position_size, avg_cost)

            # Process potential buy signals
            elif position_size == 0 and not self.position_closed[symbol]:
                # Check if we've reached maximum positions
                current_position_count = len([p for p in active_positions if active_positions[p]['size'] > 0])

                if current_position_count >= self.p.max_positions:
                    logger.info(f"Maximum position count reached ({self.p.max_positions}). Skipping buy signal evaluation.")
                else:
                    logger.info(f"Evaluating buy signal for {symbol}")


                    #commented out for testing 
                    #self.process_buy_signal(data)

            # Handle unexpected short positions
            elif position_size < 0:
                logger.error(f"{symbol}: ERROR - Unexpected short position detected! Size: {position_size}")
                self.close_position(data, "Unexpected short position")

        # Periodically verify data status
        if self.barCounter % 10 == 0:
            self.verify_data_status()
                









    def evaluate_sell_conditions(self, data, current_date, position_size=None, avg_cost=None):
        """Evaluate whether to sell a position based on strategy rules."""
        symbol = data._name
        
        # Skip if there's a pending order
        if symbol in [order_info.get('data_name') for order_info in self.order_dict.values()]:
            return
            
        # Skip if we've already marked this position as closed
        if self.position_closed[symbol]:
            return
            
        try:
            # Get trading data for this symbol
            symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol]
            
            if symbol_data.empty:
                logger.warning(f"No trading data found for {symbol}, recreating from current position")
                
                # Use provided position info or get from broker
                if avg_cost is None:
                    position = self.getposition(data)
                    avg_cost = position.price
                    position_size = position.size
                
                # Recreate trading data
                new_data = {
                    'Symbol': symbol,
                    'LastBuySignalDate': pd.Timestamp(current_date - timedelta(days=1)),
                    'LastBuySignalPrice': avg_cost,
                    'IsCurrentlyBought': True,
                    'ConsecutiveLosses': 0, 
                    'LastTradedDate': pd.Timestamp(current_date),
                    'UpProbability': 0.0,
                    'PositionSize': position_size
                }
                
                # Update trading data
                df = self.trading_data.copy()
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                write_trading_data(df, is_live=True)
                self.trading_data = df
                
                # Get the newly added data
                entry_price = avg_cost
                entry_date = current_date - timedelta(days=1)
            else:
                # Extract data from existing record
                symbol_data = symbol_data.iloc[0]
                entry_price = float(symbol_data['LastBuySignalPrice'])
                entry_date = pd.to_datetime(symbol_data['LastBuySignalDate']).date()
            
            # Get current price
            current_price = data.close[0]
            
            # Log position details
            logger.info(f"{symbol} Position Analysis:")
            logger.info(f"Current Price: ${current_price:.2f}, Entry Price: ${entry_price:.2f}")
            logger.info(f"Entry Date: {entry_date}, Current Date: {current_date}")
            logger.info(f"Days Held: {(current_date - entry_date).days}")
            
            # Check sell conditions
            should_sell_flag, reason = should_sell(
                current_price=current_price,
                entry_price=entry_price,
                entry_date=entry_date,
                current_date=current_date,
                stop_loss_percent=self.p.stop_loss_percent,
                take_profit_percent=self.p.take_profit_percent,
                position_timeout=self.p.position_timeout,
                expected_profit_per_day_percentage=self.p.expected_profit_per_day_percentage,
                verbose=True
            )
            
            if should_sell_flag:
                logger.info(f"Sell signal triggered for {symbol}: {reason}")
                self.close_position(data, reason)
            else:
                logger.info(f"No sell signal for {symbol}")
                
        except Exception as e:
            logger.error(f"Error evaluating sell conditions for {symbol}: {e}")
            logger.error(traceback.format_exc())
            





    def process_buy_signal(self, data):
        symbol = data._name
        
        if symbol in [o.get('data_name') for o in self.order_dict.values()]:
            logger.info(f"Skipping {symbol} - Order already pending")
            return
            
        buy_signals = self.trading_data[
            (self.trading_data['IsCurrentlyBought'] == False) & 
            (self.trading_data['LastBuySignalDate'].notna())
        ]
        
        if symbol not in buy_signals['Symbol'].values:
            logger.info(f"Skipping {symbol} - Not in buy signals list")
            return
            
        size = self.calculate_position_size(data)
        if size <= 0:
            logger.info(f"Skipping {symbol} - Calculated position size is zero")
            return
            
        try:
            current_close = data.close[0]
            tick_size = 0.01 if current_close >= 1.0 else 0.0001
            
            take_profit_price = round(current_close * (1 + self.p.take_profit_percent/100), 4)
            limit_price = round(current_close + tick_size, 4)
            
            if limit_price <= current_close:
                limit_price = round(current_close * 1.0001, 4)
                
        except Exception as e:
            logger.error(f"Error calculating prices for {symbol}: {e}")
            return
            
        try:
            logger.info(f"Placing bracket buy order for {symbol}: {size} shares at ${limit_price:.2f}")
            main_order = self.buy(
                data=data,
                size=size,
                exectype=bt.Order.Limit,
                price=limit_price,
                transmit=False  # Don't transmit yet
            )
            
            stop_order = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.StopTrail,
                trailpercent=self.p.trailing_stop_atr_multiple / 100,
                parent=main_order,
                transmit=False  # Don't transmit yet
            )
            limit_order = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.Limit,
                price=take_profit_price,
                parent=main_order,
                transmit=True  # This transmits all orders in the bracket
            )
            
            self.order_dict[main_order.ref] = {
                'order': main_order,
                'data_name': symbol,
                'status': 'SUBMITTED',
                'stop_child': stop_order,
                'limit_child': limit_order,
                'take_profit': take_profit_price,
                'entry_price': limit_price
            }
            logger.info(f"Bracket order placed for {symbol}: Main order {main_order.ref}, "
                       f"Stop order {stop_order.ref}, Limit order {limit_order.ref}")
        except Exception as e:
            logger.error(f"Error placing bracket order for {symbol}: {e}")
            logger.error(traceback.format_exc())
            
            if 'main_order' in locals() and main_order:
                self.cancel(main_order)
            if 'stop_order' in locals() and stop_order:
                self.cancel(stop_order)
            if 'limit_order' in locals() and limit_order:
                self.cancel(limit_order)


    def process_buy_signal(self, data):
        """Process buy signals with IB-specific order types for Canadian residents"""
        symbol = data._name
        
        # Skip if already processing this symbol
        if symbol in [o.get('data_name') for o in self.order_dict.values()]:
            return
    
        # Get position size from your existing logic
        size = self.calculate_position_size(data)
        if size <= 0:
            return
    
        # Get current price with safety checks
        current_price = data.close[0]
        if current_price <= 0:
            return
    
        # Order parameters - Will become function arguments
        take_profit_pct = 2.0  # Example: 2% target
        trail_stop_pct = 1.0   # Example: 1% trailing stop
        oca_group = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
    
        try:
            # =====================================================================
            # 1. MAIN ENTRY ORDER - Using IBKR Smart Routing with Price Improvement
            # =====================================================================
            main_order = self.buy(
                data=data,
                size=size,
                exectype=bt.Order.Limit,  # Start with limit order
                price=current_price,
                transmit=False,
                IBalgo='Adaptive',  # IB's price improvement algo
                IBalgoParams={
                    'adaptivePriority': 'Normal',
                    'adaptiveUrgency': 'Patient'
                },
                routing='SMART' if self.p.use_smart_routing else 'IBKRATS'
            )
    
            # =====================================================================
            # 2. TAKE PROFIT ORDER - Pegged to Midpoint with ATS
            # =====================================================================
            take_profit_price = round(current_price * (1 + take_profit_pct/100), 4)
            profit_order = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.Limit,
                price=take_profit_price,
                parent=main_order,
                transmit=False,
                routing='IBKRATS',  # US-only but accessible for Canadian traders
                IBparams={
                    'ocaGroup': oca_group,
                    'ocaType': 1,  # Cancel on fill
                    'tif': 'GTC'   # Good-Til-Canceled
                }
            )
    
            # =====================================================================
            # 3. TRAILING STOP - Using IB's Advanced Trailing Logic
            # =====================================================================
            trail_stop_order = self.sell(
                data=data,
                size=size,
                exectype=bt.Order.StopTrailLimit,  # Better than basic StopTrail
                parent=main_order,
                transmit=True,  # Submit all together
                trailamount=current_price * (trail_stop_pct/100),
                limitoffset=0.005,  # 0.5% limit offset from stop price
                IBparams={
                    'ocaGroup': oca_group,
                    'ocaType': 1,
                    'tif': 'GTC',
                    'usePriceMgmtAlgo': True  # Enable IB's stop price monitoring
                },
                routing='SMART'  # Accessible in Canada for US equities
            )
    
            # =====================================================================
            # Order Tracking - Add additional IB-specific metadata
            # =====================================================================
            self.order_dict[main_order.ref] = {
                'order': main_order,
                'data_name': symbol,
                'oca_group': oca_group,
                'status': 'SUBMITTED',
                'children': {
                    'profit': profit_order,
                    'stop': trail_stop_order
                },
                'params': {
                    'entry_price': current_price,
                    'take_profit_pct': take_profit_pct,
                    'trail_stop_pct': trail_stop_pct
                }
            }
    
        except Exception as e:
            self.log_error(f"Order failed for {symbol}: {str(e)}")
            self.cancel_orders_for_symbol(symbol)
    












            
    def calculate_position_size(self, data):
        """Calculate position size using the position sizer."""
        account_value = self.broker.getvalue()
        cash = self.broker.getcash()
        price = data.close[0]
        
        # Get ATR or use a default value
        atr = self.inds[data]['atr'][0] if data in self.inds else price * 0.02
        
        return self.position_sizer.calculate_position_size(
            account_value=account_value,
            cash=cash,
            price=price,
            atr=atr,
            max_positions=self.p.max_positions
        )
    
    def close_position(self, data, reason):
        """Close a position and log the reason."""
        symbol = data._name
        logger.info(f"Closing position for {symbol} ({reason})")
        
        try:
            # Find and cancel any existing trailing stops or take profits
            for order_ref, order_info in list(self.order_dict.items()):
                if order_info.get('data_name') == symbol:
                    order = order_info.get('order')
                    
                    # Only cancel sell orders (protect orders)
                    if order and order.issell():
                        self.cancel(order)
                        logger.info(f"Canceled existing order {order_ref} for {symbol}")
                    
                    # Also cancel child orders if this is a parent
                    if 'stop_child' in order_info and order_info['stop_child']:
                        self.cancel(order_info['stop_child'])
                    if 'limit_child' in order_info and order_info['limit_child']:
                        self.cancel(order_info['limit_child'])
                    
                    # Remove from dictionary
                    del self.order_dict[order_ref]
            
            # Submit close order
            close_order = self.close(data=data)
            logger.info(f"Submitted close order for {symbol}: {close_order.ref}")
            
            # Mark as closed
            self.position_closed[symbol] = True
            
            # Update trading data
            df = read_trading_data(is_live=True)
            if symbol in df['Symbol'].values:
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(datetime.now())
                write_trading_data(df, is_live=True)
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            logger.error(traceback.format_exc())
    
    def synchronize_positions(self):
        """
        Synchronize internal position tracking with actual IB positions.
        Returns a dictionary of active positions.
        """
        try:
            ib_positions = {}
            
            # Get positions from IB
            for pos in self.broker.store.ib.positions():
                symbol = pos.contract.symbol
                size = pos.position
                avg_cost = pos.avgCost
                
                if size != 0:  # Only track non-zero positions
                    ib_positions[symbol] = {
                        'size': size,
                        'avg_cost': avg_cost
                    }
            
            # Compare with our internal tracking and fix discrepancies
            df = read_trading_data(is_live=True)
            
            # Case 1: IB has position but it's not marked as bought in our data
            for symbol, pos_info in ib_positions.items():
                symbol_data = df[df['Symbol'] == symbol]
                
                if symbol_data.empty:
                    # Add new position to our tracking
                    new_data = {
                        'Symbol': symbol,
                        'LastBuySignalDate': pd.Timestamp.now() - timedelta(days=1),
                        'LastBuySignalPrice': pos_info['avg_cost'],
                        'IsCurrentlyBought': True,
                        'ConsecutiveLosses': 0,
                        'LastTradedDate': pd.NaT,
                        'UpProbability': 0.0,
                        'PositionSize': pos_info['size']
                    }
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                    logger.info(f"Added missing position for {symbol} to tracking data")
                    
                elif not symbol_data.iloc[0]['IsCurrentlyBought']:
                    # Update existing record
                    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = pos_info['size']
                    df.loc[df['Symbol'] == symbol, 'LastBuySignalPrice'] = pos_info['avg_cost']
                    logger.info(f"Updated {symbol} to reflect actual position")
                    
                elif symbol_data.iloc[0]['PositionSize'] != pos_info['size']:
                    # Update position size if different
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = pos_info['size']
                    logger.info(f"Updated position size for {symbol}: {pos_info['size']}")
            
            # Case 2: Symbol marked as bought in our data but no position in IB
            bought_symbols = df[df['IsCurrentlyBought'] == True]['Symbol'].tolist()
            for symbol in bought_symbols:
                if symbol not in ib_positions:
                    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
                    df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp.now()
                    logger.info(f"Marked {symbol} as closed (not present in IB positions)")
            
            # Write updated data
            write_trading_data(df, is_live=True)
            self.trading_data = df
            
            logger.info(f"Position synchronization complete. Active positions: {len(ib_positions)}")
            return ib_positions
            
        except Exception as e:
            logger.error(f"Error synchronizing positions: {e}")
            logger.error(traceback.format_exc())
            return {}
    
    def stop(self):
        """Called when the strategy is stopped."""
        try:
            # Log final portfolio value
            portfolio_value = self.broker.getvalue()
            logger.info(f'Final Portfolio Value: ${portfolio_value:.2f}')
            
            # Final position sync
            self.synchronize_positions()
            
            # Log all current positions
            positions = []
            for data in self.datas:
                position = self.getposition(data)
                if position.size != 0:
                    positions.append(f"{data._name}: {position.size} shares at ${position.price:.2f}")
            
            if positions:
                logger.info(f"Positions at exit: {', '.join(positions)}")
            else:
                logger.info("No positions at exit.")
                
        except Exception as e:
            logger.error(f"Error in stop method: {e}")
            
        finally:
            super().stop()


def finalize_positions_sync(ib, trading_data_path='_Live_trades.parquet'):
    """
    Fetch IB's official open positions and update local data to reflect any mismatches.
    """
    try:
        # 1. Get real open positions from IB
        ib_positions = ib.positions()
        real_positions = {}  # e.g., { 'AAPL': 100, 'TSLA': 50, ... }
        for pos in ib_positions:
            if pos.position != 0:
                symbol = pos.contract.symbol
                size = pos.position
                real_positions[symbol] = size
        
        # 2. Read your local trades DataFrame
        df = pd.read_parquet(trading_data_path)

        # Ensure required columns exist
        required_columns = {'Symbol', 'IsCurrentlyBought'}
        if not required_columns.issubset(df.columns):
            logger.warning("DataFrame missing required columns (Symbol, IsCurrentlyBought).")
        
        # Convert local positions to a dictionary for easy comparison
        local_positions = (
            df[df['IsCurrentlyBought'] == True]
            .set_index('Symbol')['PositionSize']
            .to_dict()
        )  # e.g., { 'AAPL': 100, 'TSLA': 50 }

        # 3. Reconcile differences
        # Case A: Symbol in IB but not in local -> Mark as bought
        for symbol, real_size in real_positions.items():
            if symbol not in local_positions:
                logger.info(f"{symbol} is open in IB but not marked locally; updating local data.")
                new_data = {
                    'Symbol': symbol,
                    'IsCurrentlyBought': True,
                    'PositionSize': real_size,
                    'LastBuySignalDate': pd.Timestamp.now(),
                    'LastBuySignalPrice': 0,  # Adjust as needed
                }
                # Append or update the DataFrame
                if symbol not in df['Symbol'].values:
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                else:
                    for col, val in new_data.items():
                        df.loc[df['Symbol'] == symbol, col] = val
            else:
                local_size = local_positions[symbol]
                if local_size != real_size:
                    logger.info(
                        f"Mismatch for {symbol}: local size={local_size}, IB size={real_size}. Correcting local data."
                    )
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = real_size
        
        # Case B: Symbol in local but not in IB -> Mark as closed
        for symbol, local_size in local_positions.items():
            if symbol not in real_positions:
                logger.info(f"{symbol} is locally open but not in IB; marking as closed in local data.")
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp.now()

        # 4. Save the updated DataFrame
        df.to_parquet(trading_data_path, index=False)
        logger.info("Final data sync completed successfully. Local data now matches IB reality.")
    
    except Exception as e:
        logger.error(f"Error in finalize_positions_sync: {e}")
        logger.error(traceback.format_exc())














def create_ib_connection(host='127.0.0.1', port=7497, max_attempts=3, timeout=20.0, debug=True):
    """
    Create a robust IB connection with proper disconnection handling.
    """
    dprint = globals().get('dprint', print)
    dprint("Starting IB connection process")
    
    for attempt in range(1, max_attempts + 1):
        # Generate a unique client ID for each attempt
        client_id = random.randint(10000, 99999)
        dprint(f"Connection attempt {attempt}/{max_attempts} with clientId={client_id}")
        
        try:
            # First try a direct connection to verify IB is responsive
            test_ib = ibi.IB()
            test_ib.connect(host, port, clientId=client_id, readonly=True, timeout=timeout/2)
            
            if test_ib.isConnected():
                dprint(f"Test connection successful with clientId={client_id}")
                server_time = test_ib.reqCurrentTime()
                dprint(f"Server time: {server_time}")
                
                # Disconnect test connection
                test_ib.disconnect()
                dprint("Test connection disconnected")
                
                # Now create the real connection via IBStore
                dprint(f"Creating IBStore with clientId={client_id}")
                # Use backtrader's built-in IBStore
                
                store = IBStore(
                    host=host,
                    port=port,
                    clientId=client_id,
                    reconnect=3,
                    timeout=timeout,
                    notifyall=False,
                )
                
                # Get IB instance from store
                ib = store.ib
                
                # Verify the store connection is working
                if ib.isConnected():
                    dprint(f"IBStore connection successful")
                    return store, ib
                else:
                    dprint("IBStore connection failed")
                    if store:
                        try:
                            store.stop()
                        except:
                            pass
            else:
                dprint("Test connection failed - IB may not be running or accessible")
                
        except Exception as e:
            dprint(f"Connection error: {type(e).__name__}: {e}")
            
            # Clean up any resources
            if 'test_ib' in locals() and test_ib.isConnected():
                try:
                    test_ib.disconnect()
                except:
                    pass
                    
            if 'store' in locals() and store:
                try:
                    store.stop()
                except:
                    pass
                
        # Wait before next attempt
        if attempt < max_attempts:
            dprint(f"Waiting before next connection attempt...")
            time.sleep(2)
    
    dprint(f"Failed to connect after {max_attempts} attempts")
    return None, None









def disconnect_ib_safely(store, ib, debug=True):
    """
    Safely disconnect from IB with proper cleanup to avoid hanging connections.
    Uses the simplified approach that works in your test script.
    
    Args:
        store: IBStore instance
        ib: IB instance
        debug (bool): Enable debug output
        
    Returns:
        bool: True if disconnection was attempted
    """
    dprint = globals().get('dprint', print)  # Use dprint if available, otherwise use print
    
    if debug:
        dprint("Starting IB disconnection process")
    
    # First disconnect the IB instance if it exists and is connected
    if ib:
        try:
            if ib.isConnected():
                if debug:
                    dprint("Disconnecting IB instance")
                ib.disconnect()
                if debug:
                    dprint("IB disconnect() called")
        except Exception as e:
            if debug:
                dprint(f"Error during IB disconnect: {type(e).__name__}: {e}")
    
    # Then stop the store if it exists
    if store:
        try:
            if debug:
                dprint("Stopping IBStore")
            store.stop()
            if debug:
                dprint("IBStore stop() called")
        except Exception as e:
            if debug:
                dprint(f"Error stopping IBStore: {type(e).__name__}: {e}")
    
    return True




















def start(manual_override=False):
    """
    Main entry point for the trading session with proper data feed configuration.
    
    Args:
        manual_override (bool): Whether to bypass market open checks
    """
    dprint = globals().get('dprint', print)
    
    start_time = datetime.now()
    dprint(f"===== Entering start() at {start_time.isoformat()} =====")
 
    cerebro = bt.Cerebro()
    dprint("Created Cerebro instance")
    
    # Create connection with proper client ID management
    store, ib = create_ib_connection(
        port=7497,
        max_attempts=3,
        timeout=5.0
    )
    
    if not store or not ib or not ib.isConnected():
        dprint("[start()] CRITICAL: Could not establish connection to IB. Exiting.")
        return

    dprint("[start()] Connection established successfully. Proceeding...")
    
    try:
        # Check market status if not overridden
        if not manual_override:
            if not wait_for_market_open(manual_override=manual_override):
                dprint("[start()] Market not open. Exiting.")
                disconnect_ib_safely(store, ib)
                return
        
        # Get open positions
        open_positions = get_open_positions(ib)
        dprint(f"[start()] Found {len(open_positions)} open positions")
        
        # Get buy signals
        buy_signals = get_buy_signals(is_live=True) or []
        if not buy_signals:
            buy_signals = get_buy_signals(is_live=False) or []
        dprint(f"[start()] Found {len(buy_signals)} buy signals")
        
        # Combine symbols from positions and signals
        all_symbols = set(open_positions)
        symbol_signals = [signal.get('Symbol') for signal in buy_signals if signal and isinstance(signal, dict) and 'Symbol' in signal]
        all_symbols.update(symbol_signals)
        
        if not all_symbols:
            dprint("[start()] No symbols to trade. Exiting.")
            disconnect_ib_safely(store, ib)
            return
        
        dprint(f"[start()] Will process {len(all_symbols)} total symbols: {', '.join(all_symbols)}")
        
        # Add data feeds for each symbol with proper configuration
        for symbol in all_symbols:
            try:
                # Create contract
                contract = ibi.Stock(symbol, 'SMART', 'USD')
                dprint(f"[start()] Creating contract for {symbol}")
                
                # Configure and add the data feed using RTBars (5-second bars)
                data = store.getdata(
                    dataname=symbol,
                    contract=contract,
                    sectype='STK',  # Explicit stock type
                    exchange='SMART',
                    currency='USD',
                    rtbar=True,  # Use 5-second RealTimeBars
                    timeframe=bt.TimeFrame.Seconds,  # Must be Seconds for rtbar
                    compression=5,  # 5-second bars (minimum for rtbar)
                    useRTH=True,  # Regular trading hours only
                    qcheck=0.5,  # Check for new bars every half second
                    backfill_start=False,  # Don't request historical backfill
                    backfill=False,
                    reconnect=True,
                    live=True,
                )
                
                # Set a name for the data feed for easier identification
                data._name = f"{symbol}_5sec"
                
                # Add the 5-second data to cerebro
                cerebro.adddata(data)
                dprint(f"[start()] Added 5-second data feed for {symbol}")
                
                # Resample to 1-minute bars
                minute_data = cerebro.resampledata(
                    data,
                    timeframe=bt.TimeFrame.Minutes,
                    compression=1,  # <- THIS WAS YOUR MAIN ISSUE (12 -> 1)
                    bar2edge=True,  # Align to minute boundaries
                    boundoff=0.5  # Allow 0.5 compression units to complete bar
                )
                
                # Set a clear name for the resampled data
                minute_data._name = symbol  # Just use the symbol name for the 1-min data
                
                dprint(f"[start()] Successfully resampled {symbol} to 1-minute bars")
                
            except Exception as e:
                dprint(f"[start()] Error adding data feed for {symbol}: {e}")
                dprint(traceback.format_exc())
        
        # Set up the broker and strategy
        broker = store.getbroker()
        cerebro.setbroker(broker)
        
        # Add the strategy
        cerebro.addstrategy(StockSniperLive)
        dprint("[start()] Added StockSniperLive strategy")
        
        # Run the strategy with a shorter max runtime
        dprint("[start()] Starting strategy execution")
        cerebro.run()
        dprint("[start()] Strategy execution completed")
        
    except Exception as e:
        dprint(f"[start()] CRITICAL: Unexpected error: {e}")
        dprint(traceback.format_exc())

    finally:
        # Proper cleanup in finally block
        try:
            # Final sync if available
            if ib and ib.isConnected():
                dprint("[start()] Performing final position synchronization")
                finalize_positions_sync(ib)
                
            # Always disconnect safely
            if store or ib:
                dprint("[start()] Disconnecting from IB...")
                success = disconnect_ib_safely(store, ib)
                dprint(f"[start()] Disconnection {'successful' if success else 'may have had issues'}")
        except Exception as e:
            dprint(f"[start()] Error during cleanup: {e}")
            
        # Log total runtime
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        dprint(f"===== Exiting start() at {end_time.isoformat()} "
              f"(total duration: {total_duration:.2f} sec) =====")















if __name__ == "__main__":
    # Set up logger
    logger = get_logger(script_name="6__DailyBroker")
    logger.info('logger initialized')
    logger.info('========== Starting new StockSniper Live trading session ==========')
    
    # Check if market is open (or use override for testing)
    manual_override = False  # Set to True for testing outside market hours
    if not is_nyse_open(manual_override=manual_override) and not manual_override:
        logger.error("Market is closed. Use manual_override=True to force execution.")
        sys.exit(1)
        
    # Start the trading system
    start(manual_override=manual_override)