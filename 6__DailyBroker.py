#!/root/root/miniconda4/envs/tf/bin/python
import backtrader as bt
from backtrader_ib_insync import IBStore
import logging
from logging.handlers import RotatingFileHandler

import ib_insync as ibi
import pandas as pd
import numpy as np
from datetime import datetime
import random  # <---------------- only used for client id's rest of the system is fully deterministic
from Trading_Functions import *
import traceback
import sys
import time
from zoneinfo import ZoneInfo

import socket
import exchange_calendars as ec



DEBUG_MODE = True 



def dprint(message):
    """Debug print function that can be toggled on/off."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
        logging.debug(message)


def setup_logging(log_to_console=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler('__BrokerLive.log', maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if log_to_console:
        # Create a console handler only if log_to_console is True
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def get_open_positions(ib):
    try:
        positions = ib.positions()
        positions_list = []
        for position in positions:
            if position.position != 0:
                contract = position.contract
                positions_list.append(contract.symbol)
                logging.info(f'Position found: {contract.symbol}, {position.position}')
        
        if not positions_list:
            logging.info('No open positions found')
        
        return positions_list
    except Exception as e:
        logging.error(f'Error fetching positions: {e}')
        return []








def check_tws_port(host='127.0.0.1', port=7497, timeout=2):
    """
    Minimal TWS port check via socket.
    Returns True if the port is open (TWS is likely running), False if not.
    """
    dprint(f"Starting TWS port check via socket at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            dprint("Port check successful. TWS is likely running.")
            return True
        else:
            dprint(f"Port {port} on {host} is not open. Check if TWS/IB Gateway is running.")
            return False
    except Exception as e:
        dprint(f"Socket error checking TWS port: {e}")
        return False



















nyse = ec.get_calendar('XNYS')


def wait_for_market_open(tz=ZoneInfo('America/New_York'), max_wait_minutes=180, manual_override=False):
    """
    Blocks execution until the NYSE is open or until max_wait_minutes is reached.
    Returns True if the market opened within that time, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        logging.info("Manual override enabled - bypassing market open check")
        return True
        
    logging.info("Checking if the market is currently open...")

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
            logging.error(f"Error fetching session times for {current_date}: {e}")
            return False

        # Check if the current time is within trading hours
        if market_open <= now <= market_close:
            logging.info("Market is open.")
            return True

    # If the market is not open, determine the next open time
    next_open = nyse.next_open(now)
    if not next_open:
        logging.warning("No upcoming market open found. Possibly a holiday.")
        return False

    logging.info(f"Waiting for the market to open at {next_open}...")

    # Calculate maximum wait time in seconds
    max_wait_seconds = max_wait_minutes * 60
    wait_seconds = 0
    interval = 30  # Check every 30 seconds

    # Wait loop
    while wait_seconds < max_wait_seconds:
        now = pd.Timestamp.now(tz)
        if now >= next_open:
            logging.info("Market has opened.")
            return True
        time.sleep(interval)
        wait_seconds += interval

    logging.warning("Market did not open within the expected wait time.")
    return False




def is_nyse_open(manual_override=False):
    """
    Returns True if the NYSE is currently open, False otherwise.
    
    Parameters:
    - manual_override: If True, bypasses all checks and returns True
    """
    if manual_override:
        logging.info("Manual override enabled - reporting market as open")
        return True
        
    tz_nyse = ZoneInfo('America/New_York')
    now_nyse = pd.Timestamp.now(tz_nyse)
    current_date = now_nyse.date()

    # Check if today is a trading day
    if not nyse.is_session(current_date):
        logging.info("Today is not a trading day.")
        return False

    # Get market open and close times for the current date
    try:
        market_open = nyse.session_open(current_date)
        market_close = nyse.session_close(current_date)
    except Exception as e:
        logging.error(f"Error fetching session times for {current_date}: {e}")
        return False

    # Check if current time is within trading hours
    if market_open <= now_nyse <= market_close:
        logging.info("NYSE is currently open.")
        return True
    else:
        logging.info("NYSE is currently closed.")
        return False












class MyStrategy(bt.Strategy):
    params = (
        ('max_positions', 4),
        ('reserve_percent', 0.4),
        ('stop_loss_percent', 5),
        ('take_profit_percent', 100),
        ('position_timeout', 9),
        ('expected_profit_per_day_percentage', 0.25),
        ('max_daily_drop_percent', 9.5),
        ('take_profit_multiplier', 2.0),
        ('debug', True),
        ('assume_live', True),  # New parameter to control this behavior
    )

    def __init__(self):
        self.rule_201_filter = set()
        self.order_dict = {}
        self.market_open = True
        self.data_ready = {d: False for d in self.datas}
        self.barCounter = 0
        self.trading_data = read_trading_data(is_live=True)  # Read live trading data
        self.position_closed = {d._name: False for d in self.datas}  # Track if a position has been closed
        self.live_trades = pd.DataFrame()  # Initialize live_trades as an empty DataFrame
        self.position_data = self.load_position_data()
        self.start_time = datetime.now()
        # Add a timer for heartbeat
        self.add_timer(
            when=bt.Timer.SESSION_START,
            offset=timedelta(minutes=0),  # No offset at the session start
            repeat=timedelta(minutes=1),  # Repeat every minute
            weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        )


    def get_current_position_count(self):
        """Count how many symbols are either actively held or have a pending buy order."""
        count = 0

        # 1) Count actual open positions from the broker
        #    This is the simplest approach: getposition() for each data feed
        for datafeed in self.datas:
            pos = self.getposition(datafeed)
            if pos.size > 0:
                count += 1

        # 2) Count any pending buy orders in self.order_dict that have not completed
        #    (Optional, depending on whether you want to count pending "Submitted" as 'open')
        for order_info in self.order_dict.values():
            order = order_info.get('order')
            status = order_info.get('status')
            data_name = order_info.get('data_name')

            # If this is a buy order and not yet filled/canceled/rejected,
            # then we treat it as a future open position. 
            # We can detect buy orders in Backtrader by order.isbuy() or by checking order_info.
            if order and order.isbuy() and status in ['Submitted', 'PreSubmitted', 'Accepted']:
                count += 1

        return count





    def load_position_data(self):
        df = read_trading_data(is_live=True)
        return {row['Symbol']: row.to_dict() for _, row in df.iterrows()}

    def initialize_order_dict(self):
        active_orders = self.fetch_active_orders()
        for order in active_orders:
            self.order_dict[order.orderId] = order
        logging.info(f"Initialized with {len(self.order_dict)} active orders.")

    def fetch_active_orders(self):
        """Fetch active orders from IB."""
        return self.ib.reqAllOpenOrders()

    def update_order_dict(self):
        """Update the order dictionary with currently active orders."""
        active_orders = self.fetch_active_orders()
        for order in active_orders:
            self.order_dict[order.orderId] = order
        
        # Remove completed or canceled orders
        for order_id in list(self.order_dict.keys()):
            if self.order_dict[order_id].status not in ['Submitted', 'PreSubmitted']:
                del self.order_dict[order_id]


    def check_market_status(self):
        return is_market_open(), None



    def safe_get_date(self):
        """Safely get current date from data feed or system"""
        try:
            return bt.num2date(self.datas[0].datetime[0]).date()
        except AttributeError:
            return datetime.now().date()
        except Exception as e:
            logging.error(f"Error getting date: {e}")
            return datetime.now().date()


    def notify_timer(self, timer, when, *args, **kwargs):
        elapsed_mins = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_mins >= 5:  # 5 minute runtime
            logging.info('Strategy runtime of 5 minutes reached. Starting shutdown sequence...')

            # Stop cerebro
            self.env.runstop()
            # Disconnect from IB
            try:
                store = self.broker.store
                if hasattr(store, 'disconnect'):
                    store.disconnect()
                    logging.info('Disconnected from Interactive Brokers')
            except Exception as e:
                logging.error(f"Error disconnecting from IB: {e}")

            # Log final message
            logging.info('Shutdown sequence complete. Exiting in 5 seconds...')

            # Wait 5 seconds then exit
            time.sleep(5)
            sys.exit(0)

        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f'Heartbeat: {current_time} - Running for {elapsed_mins:.1f} minutes')





    def debug(self, msg):
        if self.p.debug:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DEBUG] {current_time}: {msg}")
            logging.debug(msg)


    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        print(f"Data status change for {data._name}: {data._getstatusname(status)}")
        if status == data.LIVE:
            logging.info(f"{data._name} is now live.")
        elif status == data.DISCONNECTED:
            logging.warning(f"{data._name} disconnected.")
        elif status == data.DELAYED:
            logging.info(f"{data._name} is in delayed mode.")



    def notify_order(self, order):
        """Updated order notification handler with proper dictionary structure"""
        # Handle submitted or accepted orders
        if order.status in [order.Submitted, order.Accepted]:
            # Store the order in the dictionary with proper structure
            self.order_dict[order.ref] = {
                'order': order,
                'data_name': order.data._name,
                'status': order.getstatusname()
            }
            logging.info(f"Order for {order.data._name} {order.getstatusname()}")
            return

        # Handle completed orders
        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order)
            elif order.issell():
                self.handle_sell_order(order)
                self.position_closed[order.data._name] = True
            
            logging.info(f"Order for {order.data._name} {order.getstatusname()}")

        # Handle failed orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f'Order {order.data._name} failed with status: {order.getstatusname()}')

        # Remove completed or failed orders from dictionary
        if order.ref in self.order_dict:
            del self.order_dict[order.ref]


    def prenext(self):
        """Pre-calculate Rule 201 status for all symbols"""
        for d in self.datas:
            if len(d) > 1:
                prev_close = d.close[-1]
                current_price = d.close[0]
                drop_pct = ((prev_close - current_price) / prev_close) * 100
                if drop_pct > self.params.max_daily_drop_percent:
                    self.rule_201_filter.add(d._name)
                    logging.info(f"Rule 201 filter triggered for {d._name} ({drop_pct:.2f}% drop)")





    def next(self):
        self.barCounter += 1
        current_date = self.safe_get_date()
        self.trading_data = read_trading_data(is_live=True)  # Read live trading data
        self.live_trades = self.trading_data  # Update live_trades with the latest data

        logging.info(f"Bar {self.barCounter}: Current date: {current_date}")
        logging.info(f"Live trades data shape: {self.live_trades.shape}")
        logging.info(f"Live trades columns: {self.live_trades.columns}")

        self.market_open, _ = self.check_market_status()

        if not self.market_open:
            logging.info("Market is closed. Disconnecting from Cerebro.")
            self.stop()
            return

        print(f"Bar {self.barCounter}: Open: {self.data.open[0]}, High: {self.data.high[0]}, Low: {self.data.low[0]}, Close: {self.data.close[0]}, Volume: {self.data.volume[0]}")

        # Print currently owned stocks and their purchase dates
        current_positions = self.live_trades[self.live_trades['IsCurrentlyBought'] == True] if 'IsCurrentlyBought' in self.live_trades.columns else pd.DataFrame()
        if not current_positions.empty:
            print("\nCurrently owned stocks:")
            for _, position in current_positions.iterrows():
                symbol = position['Symbol']
                purchase_date = position['LastBuySignalDate']
                print(f"Symbol: {symbol}, Purchase Date: {purchase_date}")

        print(f"Number of data feeds: {len(self.datas)}")

        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            logging.info(f"{symbol} - Position size: {position.size}, Closed flag: {self.position_closed[symbol]}")

            if position.size > 0 and not self.position_closed[symbol]:
                logging.info(f"{symbol} - Current position size: {position.size} at average price: {position.price}")
                self.evaluate_sell_conditions(data, current_date)
            elif position.size == 0 and self.position_closed[symbol]:
                self.position_closed[symbol] = False
                logging.info(f"{symbol} - Position closed and flag reset")
            elif position.size == 0 and not self.position_closed[symbol]:
                logging.info(f'{symbol}: No position flag detected fresh check.')
                print(f'{symbol}: No position. Evaluating buy signals for data.')
                if self.get_current_position_count(self) > self.params.max_positions:
                    logging.info("Max position count reached. Skipping buy signal evaluation.")
                    continue
                else:
                    logging.info(f'Correct amount of symbols detected evaluating buy signals for symbol {symbol}')
                    self.process_buy_signal(data)
                
            elif position.size < 0:
                logging.error(f'{symbol}: ERROR - We have shorted this stock! Current position size: {position.size}')
                self.close_position(data, "Short position detected")

             
    ##============[EVALUATE SELL CONDITIONS]==================##
    ##============[EVALUATE SELL CONDITIONS]==================##
    ##============[EVALUATE SELL CONDITIONS]==================##
    
    def evaluate_sell_conditions(self, data, current_date):
        """Evaluate whether to sell a position with improved data handling"""
        symbol = data._name
        if symbol in self.order_dict or self.position_closed[symbol]:
            return

        try:
            # Read the latest trading data
            self.trading_data = read_trading_data(is_live=True)

            # Get symbol data with better error handling
            symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol]
            if symbol_data.empty:
                logging.warning(f"No trading data found for {symbol}, attempting to recreate from current position")

                # Recreate trading data from current position
                position = self.getposition(data)
                current_data = {
                    'Symbol': symbol,
                    'LastBuySignalDate': pd.Timestamp(current_date - timedelta(days=1)),  # Assume bought yesterday if unknown
                    'LastBuySignalPrice': position.price,
                    'IsCurrentlyBought': True,
                    'ConsecutiveLosses': 0,
                    'LastTradedDate': pd.Timestamp(current_date),
                    'UpProbability': 0.0,
                    'PositionSize': position.size
                }

                # Update the trading data
                self.trading_data = pd.concat([self.trading_data, pd.DataFrame([current_data])], ignore_index=True)
                write_trading_data(self.trading_data, is_live=True)
                symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
            else:
                symbol_data = symbol_data.iloc[0]

            # Get all the necessary values for sell evaluation
            entry_price = float(symbol_data['LastBuySignalPrice'])
            entry_date = pd.to_datetime(symbol_data['LastBuySignalDate']).date()
            current_price = data.close[0]
            current_date = pd.to_datetime(current_date).date()

            logging.info(f"{symbol} Position Analysis:")
            logging.info(f"Current Price: {current_price}, Entry Price: {entry_price}")
            logging.info(f"Entry Date: {entry_date}, Current Date: {current_date}")
            logging.info(f"Days Held: {(current_date - entry_date).days}")

            if should_sell(current_price, entry_price, entry_date, current_date, 
                          self.params.stop_loss_percent, self.params.take_profit_percent, 
                          self.params.position_timeout, self.params.expected_profit_per_day_percentage, verbose=True):
                logging.info(f"Sell conditions met for {symbol}. Initiating close_position.")
                self.close_position(data, "Sell conditions met")
                return
            else:
                logging.info(f"Sell conditions not met for {symbol}.")

        except Exception as e:
            logging.error(f"Error evaluating sell conditions for {symbol}: {e}")
            logging.error(traceback.format_exc())



    ###================[Update system to see orders on restart]=================#
    def handle_buy_order(self, order):
        """Handle completed buy orders with proper trade data updating"""
        symbol = order.data._name
        entry_price = order.executed.price
        entry_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logging.info(f"Buy order for {symbol} completed. Size: {position_size}, Price: {entry_price}")

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
            logging.info(f"Successfully updated trade data for buy order: {symbol}")

            # Update active positions set
            if not hasattr(self, 'active_positions'):
                self.active_positions = set()
            if symbol not in self.active_positions:
                self.active_positions.add(symbol)

        except Exception as e:
            logging.error(f"Error updating trade data for {symbol}: {e}")
            logging.error(traceback.format_exc())









    def handle_sell_order(self, order):
        symbol = order.data._name
        exit_price = order.executed.price
        exit_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logging.info(f"Sell order for {symbol} completed. Size: {position_size}, Price: {exit_price}")

        # Update the Parquet file
        update_trade_data(symbol, 'sell', exit_price, exit_date, 0, is_live=True)

        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            self.active_positions.remove(symbol)
            logging.info(f"Removed {symbol} from active positions after sell order completion")


    def calculate_position_size(self, data):
        total_value = self.broker.getvalue()
        cash_available = self.broker.getcash()
        workable_capital = total_value * (1 - self.params.reserve_percent)
        capital_for_position = workable_capital / self.params.max_positions
        size = int(capital_for_position / data.close[0])
        if cash_available < (total_value * self.params.reserve_percent) or cash_available < capital_for_position:
            return 0
        return size












    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#



    def process_buy_signalNOTAJUSTEDTOCONTRACT(self, data):
        """Process buy signals with proper price rounding and order handling"""
        symbol = data._name
        logging.info(f"Processing buy signal for {symbol}")

        # Check for existing orders/positions using the correct dictionary structure
        if symbol in [order_info.get('data_name') for order_info in self.order_dict.values()]:
            logging.info(f"Order already pending for {symbol}, skipping buy signal")
            return

        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            logging.info(f"{symbol} is already in active positions, skipping buy signal")
            return

        currently_bought = self.live_trades[self.live_trades['IsCurrentlyBought'] == True]

        if symbol not in currently_bought['Symbol'].values:
            size = self.calculate_position_size(data)
            if size > 0:
                try:
                    # Get current price and round it properly
                    current_price = data.close[0]
                    # Round to 2 decimal places for most stocks, 4 for penny stocks
                    tick_size = 0.01 if current_price >= 1.0 else 0.0001
                    limit_price = round(current_price * 1.001 / tick_size) * tick_size

                    # Create main market buy order with rounded price
                    main_order = self.buy(
                        data=data,
                        size=size,
                        exectype=bt.Order.Limit,
                        price=limit_price,
                        transmit=True  # Changed to True to ensure order is sent
                    )

                    # Only create trailing stop if main order is valid
                    if main_order and main_order.ref:
                        # Create trailing stop order
                        trail_stop = self.sell(
                            data=data,
                            size=size,
                            exectype=bt.Order.StopTrail,
                            trailpercent=0.03,  # 3% trailing stop
                            parent=main_order,
                            transmit=True
                        )

                        # Store orders in dictionary only if both are created successfully
                        if trail_stop and trail_stop.ref:
                            self.order_dict[main_order.ref] = {
                                'order': main_order,
                                'trail_stop': trail_stop,
                                'data_name': symbol,
                                'status': 'SUBMITTED'
                            }
                            logging.info(f"Orders placed successfully for {symbol} at limit price {limit_price}")
                        else:
                            logging.error(f"Failed to create trailing stop order for {symbol}")
                    else:
                        logging.error(f"Failed to create main order for {symbol}")

                except Exception as e:
                    logging.error(f"Error placing orders for {symbol}: {str(e)}")
                    # Cancel any partial orders if there was an error
                    if 'main_order' in locals() and main_order:
                        self.cancel(main_order)
                    if 'trail_stop' in locals() and trail_stop:
                        self.cancel(trail_stop)




    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##







    def place_bracket_buy(self, data, size, limit_price, stop_trailpercent, take_profit_price):
        """
        Manually create a 3-order bracket:
          1) Main buy at limit_price (transmit=False)
          2) Low-side trailing stop (transmit=False, parent=main)
          3) High-side limit (transmit=True, parent=main)

        If the main buy is filled, the children become active.
        If the trailing stop is hit, the take-profit is canceled (OCO).
        If the take-profit is hit, the trailing stop is canceled (OCO).
        If the main buy expires/cancels, the children also cancel automatically.
        """
        # -- Main Limit Buy Order (parent, transmit=False) --
        main_order = self.buy(
            data      = data,
            size      = size,
            exectype  = bt.Order.Limit,
            price     = limit_price,
            transmit  = False  # Do NOT transmit yet, so we can attach children
        )
        dprint(f"Main buy order created for {data._name} @ {limit_price} (ref={main_order.ref})")

        # -- Low-Side Trailing Stop (child, transmit=False) --
        stop_order = self.sell(
            data          = data,
            size          = size,
            exectype      = bt.Order.StopTrail,
            trailpercent  = stop_trailpercent,   # e.g. 0.03 = 3% below
            parent        = main_order,
            transmit      = False  # Wait to transmit with the last child
        )
        dprint(f"Trailing stop order created @ {stop_trailpercent*100:.1f}% below (ref={stop_order.ref})")

        # -- High-Side Take-Profit Limit (child, transmit=True) --
        # Setting transmit=True triggers sending *all* orders (the bracket).
        limit_order = self.sell(
            data      = data,
            size      = size,
            exectype  = bt.Order.Limit,
            price     = take_profit_price,
            parent    = main_order,
            transmit  = True  # Transmit everything now
        )
        dprint(f"Take-profit limit order created @ {take_profit_price} (ref={limit_order.ref})")

        # Keep references together if desired
        return main_order, stop_order, limit_order






    def process_buy_signal(self, data):
        """Enhanced with Rule 201 filtering and profit management"""
        symbol = data._name
        
        # Skip if in Rule 201 filtered set
        if symbol in self.rule_201_filter:
            logging.info(f"Skipping {symbol} - exceeds max daily drop threshold")
            return

        # Existing duplicate check
        current_order_symbols = [o.get('data_name') for o in self.order_dict.values()]
        if symbol in current_order_symbols:
            return

        # Existing ownership check
        currently_bought = self.live_trades[self.live_trades['IsCurrentlyBought'] == True]
        if symbol in currently_bought['Symbol'].values:
            return

        # Position sizing (preserve your existing logic)
        size = self.calculate_position_size(data)
        if size <= 0:
            return

        # Enhanced price calculations
        try:
            current_close = data.close[0]
            tick_size = 0.01 if current_close >= 1.0 else 0.0001
            
            # Preserve your 100% take-profit logic
            take_profit_price = round(current_close * self.params.take_profit_multiplier, 4)
            limit_price = round(current_close + tick_size, 4)
            
            # Safety check for limit price
            if limit_price <= current_close:
                limit_price = round(current_close * 1.0001, 4)

        except IndexError:
            logging.error(f"Data not available for {symbol}")
            return

        # Place bracket order if all checks pass
        main_order, stop_order, limit_order = self.place_bracket_buy(
            data=data,
            size=size,
            limit_price=limit_price,
            stop_trailpercent=0.03,
            take_profit_price=take_profit_price
        )

        if main_order and main_order.ref:
            self.order_dict[main_order.ref] = {
                'order': main_order,
                'data_name': symbol,
                'status': 'SUBMITTED',
                'stop_child': stop_order,
                'limit_child': limit_order,
                'take_profit': take_profit_price,
                'entry_price': limit_price
            }
            logging.info(f"Bracket order placed for {symbol} | Target: {take_profit_price} (+{100*(self.params.take_profit_multiplier-1):.0f}%)")















    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##

    def close_position(self, data, reason):
        """Close position and cancel associated trailing stop."""
        symbol = data._name
        logging.info(f"Attempting to close position for {symbol} due to: {reason}")

        # Find and cancel trailing stop first
        for order_ref, order_info in list(self.order_dict.items()):
            if order_info['data_name'] == symbol:
                if 'trail_stop' in order_info and order_info['trail_stop']:
                    self.cancel(order_info['trail_stop'])
                    logging.info(f"Cancelled trailing stop for {symbol}")
                del self.order_dict[order_ref]

        # Close the position
        order = self.close(data=data)
        logging.info(f"Closing order submitted for {symbol}")


    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')
        super().stop()













##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##



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
            logging.warning("DataFrame missing required columns (Symbol, IsCurrentlyBought).")
        
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
                logging.info(f"{symbol} is open in IB but not marked locally; updating local data.")
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
                    logging.info(
                        f"Mismatch for {symbol}: local size={local_size}, IB size={real_size}. Correcting local data."
                    )
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = real_size
        
        # Case B: Symbol in local but not in IB -> Mark as closed
        for symbol, local_size in local_positions.items():
            if symbol not in real_positions:
                logging.info(f"{symbol} is locally open but not in IB; marking as closed in local data.")
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp.now()

        # 4. Save the updated DataFrame
        df.to_parquet(trading_data_path, index=False)
        logging.info("Final data sync completed successfully. Local data now matches IB reality.")
    
    except Exception as e:
        logging.error(f"Error in finalize_positions_sync: {e}")
        logging.error(traceback.format_exc())




























###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###











def start(manual_override=False):
    """
    Main entry point for the trading session.
    Uses IBStore as the sole connection to IB/TWS,
    avoiding duplicate connections and clientId conflicts.
    """
    start_time = datetime.now()
    dprint(f"===== Entering start() at {start_time.isoformat()} =====")

    cerebro = bt.Cerebro()

 
    store = None
    ib = None
    try:
        # ============================== Create IBStore Once ==============================
        dprint("[start()] Creating IBStore instance (this will connect to TWS).")
        #
        # clientId=None lets IBStore pick a random clientId,
        # or you can specify your own if you prefer:
        #
        #   clientId=1234,
        #   host='127.0.0.1',
        #   port=7497,
        #   etc.
        #
        # See docstring for more arguments, e.g. reconnect=-1, timeoffset=True, etc.
        #
        store = IBStore(
            host='127.0.0.1',
            port=7497,
            clientId=None,   # Let IBStore pick a random client ID
            reconnect=3,     # Attempt reconnect if needed
            timeout=3.0,     # Timeout between reconnect attempts
            notifyall=False, # Only error messages
            _debug=False     # Turn on/off debug prints from IBStore
        )

        # The store instance has an `ib` attribute that is the actual ib_insync.IB object
        ib = store.ib

        # Confirm store/ib is actually connected
        if not ib.isConnected():
            dprint("[start()] CRITICAL: IBStore could not connect to TWS. Exiting.")
            return

        dprint("[start()] IBStore created and connected successfully. Proceeding...")

        # ============================== Data Collection Phase ==============================
        try:
            dprint("[start()] Gathering open positions from IB...")
            open_positions = get_open_positions(ib)  # your function uses ib.positions()
            dprint(f"[start()] Open positions found: {open_positions}")

            dprint("[start()] Gathering buy signals (backtesting)...")
            buy_signals_backtesting = get_buy_signals(is_live=False)
            dprint(f"[start()] Backtesting buy signals: {buy_signals_backtesting}")

            dprint("[start()] Gathering buy signals (live)...")
            buy_signals_live = get_buy_signals(is_live=True)
            dprint(f"[start()] Live buy signals: {buy_signals_live}")

            buy_signals = buy_signals_backtesting + buy_signals_live
            all_symbols = set(open_positions + [s.get('Symbol') for s in buy_signals if 'Symbol' in s])
            dprint(f"[start()] Found {len(all_symbols)} unique symbol(s) from positions and signals.")

            if not all_symbols:
                dprint("[start()] No symbols found to trade. Exiting early.")
                return

            for symbol in sorted(all_symbols):
                sources = []
                if symbol in open_positions:
                    sources.append("open position")
                if symbol in [s.get('Symbol') for s in buy_signals_backtesting]:
                    sources.append("backtesting signal")
                if symbol in [s.get('Symbol') for s in buy_signals_live]:
                    sources.append("live signal")
                dprint(f"[start()] Symbol '{symbol}' found in: {', '.join(sources)}")

            dprint("[start()] Data collection phase completed successfully.")

        except Exception as data_err:
            dprint(f"[start()] Error collecting trading data: {data_err}")
            dprint(traceback.format_exc())
            return

        # ============================== Data Feed Initialization ==============================
        if manual_override:
            dprint("[start()] Manual override enabled; skipping data feed initialization.")
            return
        else:
            dprint("[start()] Manual override disabled; proceeding with data feed initialization.")

        failed_symbols = []
        for symbol in sorted(all_symbols):
            try:
                # Create an ib_insync Stock contract
                contract = ibi.Stock(symbol, 'SMART', 'USD')
                
                # Get a backtrader "DataFeed" from IBStore
                data = store.getdata(
                    dataname=symbol,
                    contract=contract,
                    sectype=contract.secType,
                    exchange=contract.exchange,
                    currency=contract.currency,
                    rtbar=True,
                    what='TRADES',
                    useRTH=True,
                    qcheck=1.0,
                    backfill_start=False,
                    backfill=False,
                    reconnect=True,
                    timeframe=bt.TimeFrame.Seconds,
                    compression=5,
                    live=True,
                )

                # Optionally resample if you want a 30-second timeframe
                resampled_data = cerebro.resampledata(
                    data,
                    timeframe=bt.TimeFrame.Seconds,
                    compression=6
                )
                resampled_data._name = symbol
                dprint(f"[start()] Successfully added live data feed for '{symbol}'.")

            except Exception as feed_err:
                dprint(f"[start()] Failed to add data feed for '{symbol}': {feed_err}")
                dprint(traceback.format_exc())
                failed_symbols.append(symbol)
                continue

        if len(failed_symbols) == len(all_symbols):
            dprint("[start()] All symbol data feeds failed; cannot proceed. Exiting.")
            return

        # ============================== Run the Strategy ==============================
        try:
            broker = store.getbroker()  # IBStore also provides the broker
            cerebro.setbroker(broker)
            cerebro.addstrategy(MyStrategy)
            dprint(f"[start()] About to run Cerebro with MyStrategy on {len(all_symbols)} symbol(s).")

            cerebro.run()
            dprint("[start()] Cerebro run completed successfully.")

        except Exception as strat_err:
            dprint(f"[start()] Strategy execution failed: {strat_err}")
            dprint(traceback.format_exc())

    except Exception as e:
        dprint(f"[start()] CRITICAL: Unexpected error: {e}")
        dprint(traceback.format_exc())

    finally:
        dprint("[start()] Entering final cleanup stage...")

        # 1) Finalize positions sync if the store is connected
        try:
            if store and store.ib and store.ib.isConnected():
                dprint("[start()] Attempting to finalize position sync with TWS...")
                finalize_positions_sync(store.ib, '_Live_trades.parquet')
                dprint("[start()] Final data sync completed.")
        except Exception as sync_err:
            dprint(f"[start()] Error during final data sync: {sync_err}")
            dprint(traceback.format_exc())

        # 2) Disconnect from store.ib_insync (the store or the ib reference)
        try:
            if store and store.ib and store.ib.isConnected():
                store.ib.disconnect()
                dprint("[start()] Successfully disconnected ib_insync.")
        except Exception as disconnect_err:
            dprint(f"[start()] Error during ib_insync disconnection: {disconnect_err}")
            dprint(traceback.format_exc())

        # If you manually want to tear down the store object, you can do so,
        # but there's no method like store.isconnected() so just rely on
        # store.ib.isConnected() if needed.

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        dprint(f"===== Exiting start() at {end_time.isoformat()} "
               f"(total duration: {total_duration:.2f} sec) =====")

























if __name__ == "__main__":
    setup_logging(log_to_console=True)  # Enable console logging for testing
    logging.info('============================[ Starting new trading session ]==============')

    if is_nyse_open(manual_override=False):
        logging.info("NYSE is open (or override enabled). Starting trading operations.")
        # Increase wait time and attempts for testing
        start(manual_override=False)
    else:
        logging.error("NYSE is closed.")
        sys.exit(1)
