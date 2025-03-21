#!/usr/bin/env python 
import os
import time
import logging
import argparse
import random
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import pyarrow.parquet as pq
import multiprocessing
from numba import njit
import traceback
from collections import Counter
import pandas_market_calendars as mcal
import math
import glob
import warnings
from Util import *

from Util import (
    STRATEGY_PARAMS_TUPLE as STRATEGY_PARAMS,  # Note we're using the tuple version for backtrader
)





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
            today_market_open = today_market_open.replace(tzinfo=timezone.utc)
        
        now_utc = datetime.now(timezone.utc)
        
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

# Optimized data loading
@njit
def calculate_up_prob_variance(up_probs):
    """Calculate variance of UpProbability using numba for speed."""
    if len(up_probs) < 10:  # Need minimum sample size
        return 0.0
    return np.var(up_probs)

def filter_stocks_by_signal_quality(directory, min_variance=0.01, min_up_prob=0.5, variance_weight=1.0):

    all_files = glob.glob(os.path.join(directory, '*.parquet'))
    quality_stocks = []
    
    logging.info(f"Evaluating {len(all_files)} stocks for signal quality...")
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.starmap(
                evaluate_stock_quality, 
                [(f, min_variance, min_up_prob, variance_weight) for f in all_files]
            ),
            total=len(all_files),
            desc="Filtering stocks"
        ))
        
    stock_quality_pairs = [(r[0], r[1]) for r in results if r is not None]
    
    stock_quality_pairs.sort(key=lambda x: x[1], reverse=True)
    
    quality_stocks = [pair[0] for pair in stock_quality_pairs]
    
    if len(stock_quality_pairs) > 0:
        top_5 = stock_quality_pairs[:5]
        bottom_5 = stock_quality_pairs[-5:] if len(stock_quality_pairs) >= 5 else stock_quality_pairs
        
        logging.info("Top 5 quality stocks:")
        for file_path, score in top_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
            
        logging.info("Bottom 5 quality stocks:")
        for file_path, score in bottom_5:
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            logging.info(f"  {stock_name}: Quality Score = {score:.4f}")
    
    logging.info(f"Found {len(quality_stocks)} stocks meeting quality criteria")
    return quality_stocks

def evaluate_stock_quality(file_path, min_variance, min_up_prob, variance_weight=1.0):
    try:
        table = pq.read_table(file_path, columns=['UpProbability'])
        df = table.to_pandas()
        
        if len(df) < 60:  # Require at least 60 days of data
            return None
        
        up_probs = df['UpProbability'].values
        
        variance = calculate_up_prob_variance(up_probs)
        
        max_up_prob = np.max(up_probs)
        
        if variance < min_variance or max_up_prob < min_up_prob:
            return None
            
        norm_variance = min(variance / 0.10, 1.0)  # Cap at 1.0
        norm_max_prob = (max_up_prob - 0.5) / 0.5  # Normalize to 0-1 range
        
        quality_score = (norm_variance * variance_weight + norm_max_prob) / (1 + variance_weight)
        
        return (file_path, quality_score)
            
    except Exception as e:
        logging.error(f"Error evaluating {file_path}: {str(e)}")
    
    return None

def load_data(file_path, last_trading_date):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        yesterday = last_trading_date
        start_date = yesterday - timedelta(days=400)
        
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= yesterday)]
        
        if len(df) < 252:  # Need at least 1 year of data
            logging.info(f"Skipping {file_path} due to insufficient data: {len(df)} days")
            return None
        
        df = df.iloc[-252:]
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                          'Distance to Support (%)', 'Distance to Resistance (%)', 
                          'UpProbability', 'UpPrediction']
        
        if all(col in df.columns for col in required_columns):
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(4).astype(np.float32)
            
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            return (stock_name, df)
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            logging.warning(f"Skipping {file_path} due to missing columns: {missing_cols}")

    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        traceback.print_exc()

    return None

def parallel_load_data(file_paths, last_trading_date):
    """Load data files in parallel for better performance."""
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.starmap(load_data, [(fp, last_trading_date) for fp in file_paths]), 
            total=len(file_paths), 
            desc="Loading Files"
        ))
    
    return [result for result in results if result is not None]

def read_trading_data():
    """Read the trading data parquet file."""
    file_path = '_Buy_Signals.parquet'
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
            'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
        ])
        df.to_parquet(file_path, index=False)
        return df
    
    return pd.read_parquet(file_path)

def write_trading_data(df):
    dtype_schema = {
        'Symbol': 'string',
        'LastBuySignalPrice': 'float64',
        'IsCurrentlyBought': 'bool',
        'ConsecutiveLosses': 'int64',
        'UpProbability': 'float64',
        'LastSellPrice': 'float64',
        'PositionSize': 'float64'
    }
    
    df = df.copy()
    
    date_columns = ['LastBuySignalDate', 'LastTradedDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col, dtype in dtype_schema.items():
        if col not in df.columns:
            if dtype == 'float64':
                df[col] = pd.Series(dtype='float64')
            elif dtype == 'int64':
                df[col] = pd.Series(dtype='int64')
            elif dtype == 'bool':
                df[col] = pd.Series(dtype='bool')
            elif dtype == 'string':
                df[col] = pd.Series(dtype='string')
        else:
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
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df[col] = df[col].astype('object').where(df[col].notnull(), None)
    df.to_parquet('_Buy_Signals.parquet', index=False)

def update_buy_signal(symbol, date, price, up_probability):
    try:
        price = round(float(price), 4)
        up_probability = round(float(up_probability), 4)
        
        df = read_trading_data()
        
        new_data = pd.DataFrame([{
            'Symbol': str(symbol),
            'LastBuySignalDate': pd.Timestamp(date),
            'LastBuySignalPrice': price,
            'IsCurrentlyBought': False,
            'ConsecutiveLosses': 0,
            'LastTradedDate': pd.NaT,
            'UpProbability': up_probability,
            'LastSellPrice': float('nan'),
            'PositionSize': float('nan')
        }])
        
        # Set proper dtypes for the new DataFrame
        new_data = new_data.astype({
            'Symbol': 'string',
            'LastBuySignalPrice': 'float64',
            'IsCurrentlyBought': 'bool',
            'ConsecutiveLosses': 'int64',
            'UpProbability': 'float64',
            'LastSellPrice': 'float64',
            'PositionSize': 'float64'
        })
        
        new_data['LastBuySignalDate'] = pd.to_datetime(new_data['LastBuySignalDate'])
        new_data['LastTradedDate'] = pd.to_datetime(new_data['LastTradedDate'])
        
        df = df[df['Symbol'] != symbol]
        
        for col in new_data.columns:
            if col in df.columns and col not in ['LastBuySignalDate', 'LastTradedDate']:
                df[col] = df[col].astype(new_data[col].dtype)
        
        df = pd.concat([df, new_data], ignore_index=True)
        
        write_trading_data(df)
        
        logging.info(f"Updated buy signal for {symbol} at price {price}")
        
    except Exception as e:
        logging.error(f"Error in update_buy_signal for {symbol}: {str(e)}")
        raise

def mark_position_as_bought(symbol, position_size):
    df = read_trading_data()
    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
    df.loc[df['Symbol'] == symbol, 'PositionSize'] = position_size
    write_trading_data(df)

def update_trade_result(symbol, is_loss, exit_price=None, exit_date=None):
    df = read_trading_data()

    if symbol in df['Symbol'].values:
        if is_loss:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] += 1
        else:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] = 0
        
        df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp(exit_date or datetime.now().date())
        if exit_price is not None:
            df.loc[df['Symbol'] == symbol, 'LastSellPrice'] = exit_price
        df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
        write_trading_data(df)




class EnhancedPandasData(bt.feeds.PandasData):
    """Enhanced PandasData class that includes ML signals and technical indicators."""
    zzzlines = ('dist_to_support', 'dist_to_resistance', 'UpProbability', 'UpPrediction', 'atr')
    lines = ('UpProbability', 'atr')

    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        #('dist_to_support', 'Distance to Support (%)'),
        #('dist_to_resistance', 'Distance to Resistance (%)'),
        ('UpProbability', 'UpProbability'),
        #('UpPrediction', 'UpPrediction'),
        ('atr', None),  # Will be calculated in the strategy
    )

class FixedCommissionScheme(bt.CommInfoBase):
    params = (
        ('commission', 3.0),  # Fixed commission per trade
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission  # Return fixed commission

class Rule201Monitor:
    def __init__(self, threshold=-9.99, cooldown_days=1):
        self.threshold = threshold
        self.cooldown_days = cooldown_days
        self.violations = set()
        self.trigger_dates = {}
        self.triggerCount = 0
    
    def check_rule_201(self, symbol, prev_close, current_price, current_date):
        if prev_close <= 0:
            return False
            
        daily_return = (current_price / prev_close - 1) * 100
        
        if daily_return <= self.threshold:
            self.violations.add(symbol)
            self.trigger_dates[symbol] = current_date
            self.triggerCount += 1
            return True
        return False
    
    def clear_expired_restrictions(self, current_date):
        expired = []
        
        for symbol, trigger_date in self.trigger_dates.items():
            days_since = (current_date - trigger_date).days
            if days_since > self.cooldown_days:
                expired.append(symbol)
                logging.info(f"Rule 201 cooldown expired for {symbol}")
        
        for symbol in expired:
            self.violations.discard(symbol)
            del self.trigger_dates[symbol]
    
    def is_restricted(self, symbol):
        """Check if a symbol is currently restricted under Rule 201."""
        return symbol in self.violations

class PositionSizer:
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

class TradeRecorder:
    def __init__(self, filename='trade_history.parquet'):
        self.filename = filename
        self.trades = []
        
    def record_trade(self, trade_data):
        """Record a trade with detailed metadata."""
        self.trades.append(trade_data)
        
    def save_trades(self):
        """Save all recorded trades to a parquet file."""
        if not self.trades:
            logging.info("No trades to save")
            return
            
        df = pd.DataFrame(self.trades)
        
        numeric_cols = [
            'EntryPrice', 'ExitPrice', 'Quantity', 'PnL', 'PnLPct',
            'Commission', 'Slippage', 'ATR', 'UpProbability'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        date_cols = ['EntryDate', 'ExitDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df.to_parquet(self.filename, index=False)
        logging.info(f"Saved {len(self.trades)} trades to {self.filename}")














class StockSniperStrategy(bt.Strategy):
    # Use the parameters from STRATEGY_PARAMS_TUPLE
    params = STRATEGY_PARAMS
    
    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d]['atr'] = bt.indicators.ATR(d, period=self.p.atr_period)
            self.inds[d]['up_prob_ma3'] = bt.indicators.SMA(d.UpProbability, period=3)
            self.inds[d]['up_prob_ma5'] = bt.indicators.SMA(d.UpProbability, period=5)
            self.inds[d]['up_prob_roc'] = bt.indicators.ROC(d.UpProbability, period=3)
        
        # Rest of your initialization code remains the same
        self.order_list = []  # Track pending orders
        self.entry_prices = {}  # Track entry prices for positions
        self.position_dates = {}  # Track entry dates for positions
        self.stop_loss_orders = {}  # Track stop loss orders
        self.take_profit_orders = {}  # Track take profit orders
        self.trailing_stops = {}  # Track trailing stop levels
        self.asset_groups = {}  # Track asset groups for correlation
        self.group_allocations = {}  # Track group allocations
        
        self.trade_history = []  # Detailed trade history for analysis
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0
        self.longest_win_streak = 0
        self.longest_loss_streak = 0
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.recent_outcomes = []  # Store last 10 trade outcomes (1=win, 0=breakeven, -1=loss)
        
        self.trade_recorder = TradeRecorder('trade_history.parquet')
        
        self.open_positions = 0
        
        self.correlation_df = pd.read_parquet('Correlations.parquet')
        logging.info(f"Loaded correlation dataframe with columns: {list(self.correlation_df.columns)}")

        if 'Ticker' in self.correlation_df.columns:
            self.correlation_df_by_ticker = self.correlation_df.copy()
            self.correlation_df.set_index('Ticker', inplace=True)
            logging.info("Set 'Ticker' column as index in correlation dataframe")
        else:
            logging.warning("'Ticker' column not found in correlation dataframe. Available columns: " 
                           f"{list(self.correlation_df.columns)}")

        self.total_groups = self.correlation_df['Cluster'].nunique()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}
        
        # Use the parameters from Util for PositionSizer
        self.position_sizer = PositionSizer(
            risk_per_trade=self.p.risk_per_trade_pct,
            max_position_pct=self.p.max_position_pct,
            min_position_pct=self.p.min_position_pct,
            reserve_pct=self.p.reserve_percent
        )
        
        # Use the parameters from Util for Rule201Monitor
        self.rule_201_monitor = Rule201Monitor(
            threshold=self.p.rule_201_threshold,
            cooldown_days=self.p.rule_201_cooldown
        )
        
        self.last_trading_date = get_last_trading_date()
        self.second_last_trading_date = get_previous_trading_day(self.last_trading_date)
        self.trading_lockup_start = get_previous_trading_day(self.last_trading_date, self.p.lockup_days)
        
        self.positions_cleared_for_lockup = False
        self.trading_locked = False
        self.last_logged_date = None
        
        self.monthly_performance = {}  # {YYYY-MM: percent_return}
        self.yearly_performance = {}   # {YYYY: percent_return}
        self.last_month_equity = None
        self.last_year_equity = None
        self.month_high_equity = None
        self.month_low_equity = None
        self.current_month = None
        self.current_year = None
        
        self.day_count = 0
        self.total_bars = 252  # Expected trading days in backtest
        self.progress_bar = tqdm(
            total=self.total_bars,
            desc="Strategy Progress",
            unit="day",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )
        













    def prenext(self):
        """Run Rule 201 checks before each new trading day AKA down by more than 10% and will prevent child parent orders with trailing stops."""
        current_date = self.datetime.date()
        
        self.rule_201_monitor.clear_expired_restrictions(current_date)
        
        for d in self.datas:
            if len(d) > 1:  # Need at least 2 data points
                symbol = d._name
                prev_close = d.close[-1]
                current_price = d.open[0]
                
                self.rule_201_monitor.check_rule_201(
                    symbol, prev_close, current_price, current_date
                )
    




    def next(self):
        """Main strategy logic executed on each bar."""
        self.day_count += 1
        self.progress_bar.update(1)

        current_date = self.datetime.date()
        current_month = current_date.strftime('%Y-%m')
        current_year = current_date.strftime('%Y')

        if any(len(data) == 0 for data in self.datas):
            return

        if self.last_logged_date != current_date:
            logging.info(f"Processing date: {current_date}")
            self.last_logged_date = current_date

        current_equity = self.broker.getvalue()

        # Monthly performance tracking
        if current_month != self.current_month:
            if self.current_month is not None:
                if self.last_month_equity is not None:
                    monthly_return = (current_equity / self.last_month_equity - 1) * 100
                    self.monthly_performance[self.current_month] = monthly_return

                    logging.info(f"Month {self.current_month} performance: {monthly_return:.2f}%")
                    logging.info(f"Month high: {self.month_high_equity:.2f}, low: {self.month_low_equity:.2f}")

            self.current_month = current_month
            self.last_month_equity = current_equity
            self.month_high_equity = current_equity
            self.month_low_equity = current_equity
            logging.info(f"Starting new month: {current_month}")
        else:
            if current_equity > self.month_high_equity:
                self.month_high_equity = current_equity
            if current_equity < self.month_low_equity:
                self.month_low_equity = current_equity

        # Yearly performance tracking
        if current_year != self.current_year:
            if self.current_year is not None:
                if self.last_year_equity is not None:
                    yearly_return = (current_equity / self.last_year_equity - 1) * 100
                    self.yearly_performance[self.current_year] = yearly_return

                    logging.info(f"Year {self.current_year} performance: {yearly_return:.2f}%")

            self.current_year = current_year
            self.last_year_equity = current_equity
            logging.info(f"Starting new year: {current_year}")

        # Check if we're on the last trading date - we want to make sure we generate signals
        # but we don't need to modify the trading logic
        if current_date == self.last_trading_date:
            logging.info(f"On last trading day ({current_date}). Will generate predictions.")
            # Don't immediately generate signals - let the normal flow handle it
            # The process_buy_candidates method will handle force-selecting the best stock if needed

        # Continue with normal position management
        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        if self.open_positions < self.p.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates or current_date == self.last_trading_date:
                # This will now handle selecting the best stock if needed
                self.process_buy_candidates(buy_candidates, current_date)








    def get_buy_candidates(self, current_date):

        buy_candidates = []
        
        for d in self.datas:
            if self.getposition(d).size > 0:
                continue
                
            if self.can_buy(d, current_date):
                size = self.calculate_position_size(d)
                
                if size > 0:
                    correlation = self.get_mean_correlation(
                        d._name, 
                        [data._name for data in self.datas if self.getposition(data).size > 0]
                    )
                    buy_candidates.append((d, size, correlation))

        return buy_candidates
    
    def calculate_position_size(self, data):
        return self.position_sizer.calculate_position_size(
            account_value=self.broker.getvalue(),
            cash=self.broker.getcash(),
            price=data.close[0],
            atr=self.inds[data]['atr'][0],
            max_positions=self.p.max_positions
        )
    







    ##========================================================[BUY CONDITIONS]========================================================##

    
    def can_buy(self, data, current_date):
        """
        Enhanced can_buy function based on extensive EDA results
        Implements optimized thresholds and multi-factor decision logic
        """
        symbol = data._name
        if self.rule_201_monitor.is_restricted(symbol) or self.open_positions >= self.p.max_positions:
            return False
    
        lookback_period = min(30, len(data) - 1)
        if lookback_period < 5:  # Need at least 5 days of data
            return False
    
        current_prob = data.UpProbability[0]
        
        # Immediate rejection if probability too low - increased from original
        if current_prob < 0.55:  # Based on clear performance drop below this level
            return False
    
        try:
            # Collect probability history
            prob_array = []
            for i in range(lookback_period):
                if i < len(data):
                    prob_array.append(data.UpProbability[-i])
                else:
                    break
                    
            if len(prob_array) >= 5:
                # Calculate variance of probability values
                prob_variance = np.var(prob_array)
    
                # Reject very low variance signals (poor predictive power)
                if prob_variance < 0.0005:
                    return False
    
                # SIGNIFICANT REVISION: Higher thresholds based on EDA
                # These thresholds align with the performance heatmaps
                if prob_variance < 0.001:  # Q1 (very low variance)
                    min_prob_threshold = 0.90  # Significantly increased from 0.70
                elif prob_variance < 0.0015:  # Q2 (low variance)
                    min_prob_threshold = 0.85  # Significantly increased from 0.65
                elif prob_variance < 0.0025:  # Q3 (medium variance)
                    min_prob_threshold = 0.80  # Significantly increased from 0.60
                elif prob_variance < 0.0075:  # Q4 (medium-high variance)
                    min_prob_threshold = 0.75  # Significantly increased from 0.55
                else:  # Q5 (high variance > 0.0075)
                    min_prob_threshold = 0.70  # Significantly increased from 0.50
                    
                # Reject if below adaptive threshold
                if current_prob < min_prob_threshold:
                    return False
    
                # PREMIUM TRADE IDENTIFICATION - updated based on EDA results
                # Using the top results from the analysis
                if prob_variance > 0.005 and current_prob > 0.85:
                    high_quality_trade = True  # Top performing combination
                elif prob_variance > 0.015 and current_prob > 0.80:
                    high_quality_trade = True  # Also excellent performance
                elif prob_variance > 0.02 and current_prob > 0.75:
                    high_quality_trade = True  # Strong performance at very high variance
                else:
                    high_quality_trade = False
    
                # Trend analysis with direction requirements
                if len(prob_array) >= 5:
                    recent_direction = np.corrcoef(range(5), prob_array[:5])[0, 1]
                    
                    # Apply trend requirements - revised based on direction heatmap
                    if not high_quality_trade:
                        # Direction requirements vary by variance
                        if prob_variance < 0.001 and recent_direction < 0.2:
                            return False
                        elif prob_variance < 0.0025 and recent_direction < 0.0:
                            return False
                        elif prob_variance >= 0.0075 and recent_direction < -0.2:
                            # Note: Q5 actually showed good performance with slightly negative direction
                            return False
                    else:
                        # Premium trades should avoid strong downtrends
                        if recent_direction < -0.2:
                            return False
    
                # Consistency check - refined based on high days analysis
                if high_quality_trade:
                    # Premium trades need fewer consistent days
                    high_threshold = 0.65
                    min_high_days = 1
                elif prob_variance < 0.002:
                    # Low variance requires more consistency
                    high_threshold = 0.75  # Increased from 0.65
                    min_high_days = 3
                else:
                    # Medium/high variance
                    high_threshold = 0.70  # Increased from 0.58
                    min_high_days = 2
    
                high_prob_days = sum(1 for p in prob_array[:5] if p > high_threshold)
                if high_prob_days < min_high_days:
                    return False
    
                # Momentum check - based on momentum ratio chart
                if not high_quality_trade:
                    # Standard trades need good momentum preservation
                    if current_prob < prob_array[1] * 0.99:
                        return False
                else:
                    # Premium trades can tolerate slightly more drop
                    if current_prob < prob_array[1] * 0.97:
                        return False
                        
                # Probability acceleration check - revised based on accel chart
                if len(prob_array) >= 3:
                    # Calculate probability acceleration
                    first_diff = prob_array[0] - prob_array[1]
                    second_diff = prob_array[1] - prob_array[2]
                    acceleration = first_diff - second_diff
                    
                    # Different requirements based on quality
                    if not high_quality_trade:
                        # Standard trades work best with positive acceleration
                        if acceleration < 0 and current_prob < 0.85:
                            return False
                    else:
                        # Premium trades can accept mild deceleration
                        if acceleration < -0.02:
                            return False
    
                # Log detailed diagnostics
                variance_group = "Q1" if prob_variance < 0.001 else \
                                "Q2" if prob_variance < 0.0015 else \
                                "Q3" if prob_variance < 0.0025 else \
                                "Q4" if prob_variance < 0.0075 else "Q5"
                                
                quality_tag = "[PREMIUM]" if high_quality_trade else ""
                logging.info(f"BUY SIGNAL {quality_tag} for {symbol}: UpProb={current_prob:.3f}, Variance={prob_variance:.6f}, "
                           f"Group={variance_group}, HighDays={high_prob_days}, Direction={recent_direction:.2f}, "
                           f"Accel={acceleration:.4f}")
    
                return True
    
        except Exception as e:
            logging.warning(f"Error in can_buy for {symbol}: {str(e)}")
    
        return False
    
    












    def force_best_signal_for_current_day(self, data):
        """Find and save the best possible stock for the current day, even if it doesn't meet all criteria."""
        logging.info("No buy candidates met standard criteria. Finding best stock for current day...")

        # Create a list to store potential candidates with their quality scores
        candidates = []

        for d in self.datas:
            if self.getposition(d).size > 0:
                continue  # Skip stocks we already have positions in

            symbol = d._name
            if self.rule_201_monitor.is_restricted(symbol):
                continue  # Skip stocks restricted by Rule 201

            # Get the current UpProbability
            current_prob = d.UpProbability[0]

            # Skip really low probability stocks
            if current_prob < 0.60:
                continue

            lookback_period = min(30, len(data) - 1)
            if lookback_period < 5:  # Need at least some historical data
                continue

            try:
                # Calculate a quality score based on various factors
                prob_array = []
                for i in range(lookback_period):
                    if i < len(d):
                        prob_array.append(d.UpProbability[-i])
                    else:
                        break

                if len(prob_array) < 5:
                    continue

                # Calculate variance of probability values
                prob_variance = np.var(prob_array)

                # Calculate recent trend
                recent_direction = 0
                if len(prob_array) >= 7:
                    recent_direction = np.corrcoef(range(7), prob_array[:7])[0, 1]

                # Count high probability days
                high_days = sum(1 for p in prob_array[:7] if p > 0.65)

                # Calculate quality score (higher is better)
                # Weight current probability the most, then trend direction, then variance, then high days
                quality_score = (
                    current_prob * 50 +  # Current probability (0-50 points)
                    (recent_direction + 1) * 20 +  # Trend direction (-1 to 1) mapped to 0-40 points
                    min(prob_variance * 500, 10) +  # Variance (0-10 points)
                    high_days * 2  # High days (0-14 points)
                )

                # Add to candidates with position size calculation
                size = self.calculate_position_size(d)
                if size > 0:
                    candidates.append((d, size, 0, quality_score))  # Use 0 for correlation

            except Exception as e:
                logging.warning(f"Error evaluating forced candidate {symbol}: {str(e)}")

        # Sort candidates by quality score (highest first)
        candidates.sort(key=lambda x: x[3], reverse=True)

        if candidates:
            # Take the top candidate
            best_candidates = [candidates[0][:3]]  # Remove the quality score for compatibility

            logging.info(f"Force-selected best stock: {candidates[0][0]._name} with UpProb={candidates[0][0].UpProbability[0]:.4f}, score={candidates[0][3]:.2f}")

            # Save only this candidate as a buy signal
            self.save_best_buy_signals(best_candidates)
        else:
            logging.warning("Could not find any suitable stock for forced signal generation")











    def process_buy_candidates(self, buy_candidates, current_date):
        """Process buy candidates and execute trades."""
        if not buy_candidates:
            # MODIFICATION: If there are no buy candidates on the last trading day, find the best stock
            if current_date == self.last_trading_date:
                self.force_best_signal_for_current_day()
            return

        buy_candidates = self.sort_buy_candidates(buy_candidates)

        self.save_best_buy_signals(buy_candidates)

        for d, size, _ in buy_candidates:
            if self.open_positions < self.p.max_positions:
                if self.check_group_allocation(d):
                    self.execute_buy(d, size, current_date)
                else:
                    logging.info(f"Skipping {d._name} due to group allocation limits")
            else:
                break
    
    def sort_buy_candidates(self, buy_candidates):
        sorted_candidates = sorted(buy_candidates, key=lambda x: x[0].UpProbability[0], reverse=False)

        if sorted_candidates:
            logging.info("Top buy candidates based on UpProbability:")
            for i, (d, size, corr) in enumerate(sorted_candidates[:min(5, len(sorted_candidates))]):
                logging.info(f"  {i+1}. {d._name}: UpProb={d.UpProbability[0]:.4f}")

        return sorted_candidates



    def get_mean_correlation(self, candidate_ticker, current_positions):
        try:
            if not current_positions:
                return 0

            if candidate_ticker not in self.correlation_df.index:
                logging.warning(f"Ticker {candidate_ticker} not found in correlation data")
                return 0

            correlations = []

            candidate_row = self.correlation_df.loc[candidate_ticker]

            for pos in current_positions:
                if pos not in self.correlation_df.index:
                    continue

                position_cluster = self.correlation_df.loc[pos, 'Cluster']

                cluster_column = f"correlation_{position_cluster}"
                if cluster_column in self.correlation_df.columns:
                    corr_value = candidate_row[cluster_column]
                    correlations.append(abs(corr_value))  # Use absolute correlation

            return np.mean(correlations) if correlations else 0

        except Exception as e:
            logging.error(f"Error calculating correlations: {str(e)}")
            return 0
    



    def check_group_allocation(self, data):
        symbol = data._name

        group = None
        try:
            if hasattr(self.correlation_df, 'index') and hasattr(self.correlation_df.index, 'contains'):
                if symbol in self.correlation_df.index:
                    group = int(self.correlation_df.loc[symbol, 'Cluster'])
            elif 'Ticker' in self.correlation_df.columns:
                ticker_row = self.correlation_df[self.correlation_df['Ticker'] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
            else:
                first_col = self.correlation_df.columns[0]
                ticker_row = self.correlation_df[self.correlation_df[first_col] == symbol]
                if not ticker_row.empty:
                    group = int(ticker_row['Cluster'].iloc[0])
        except Exception as e:
            logging.warning(f"Error finding group for {symbol}: {str(e)}")

        if group is None:
            logging.info(f"No group found for {symbol}, allowing trade")
            return True  # If no group data, allow the trade

        current_allocation = self.group_allocations.get(group, 0)

        return current_allocation < self.p.max_group_allocation



    def execute_buy(self, data, size, current_date):
        symbol = data._name
        current_price = data.close[0]
        atr = self.inds[data]['atr'][0]
        
        stop_price = current_price - (atr * self.p.stop_loss_atr_multiple)
        
        take_profit = current_price * (1 + self.p.take_profit_percent / 100.0)
        
        logging.info(f"BUY {symbol}: Price={current_price:.2f}, Size={size}, "
                    f"Stop={stop_price:.2f}, Target={take_profit:.2f}, "
                    f"UpProb={data.UpProbability[0]:.4f}")
        
        buy_order = self.buy(data=data, size=size)
        
        self.order_list.append(buy_order)
        self.entry_prices[data] = current_price
        self.position_dates[data] = current_date
        self.open_positions += 1
        
        self.update_group_data(data)
        
        update_buy_signal(symbol, current_date, current_price, data.UpProbability[0])
        
        self.trailing_stops[data] = stop_price
    
    def update_group_data(self, data):
        try:
            symbol = data._name
            
            if symbol in self.correlation_df.index:
                group = int(self.correlation_df.loc[symbol, 'Cluster'])
                self.asset_groups[symbol] = group
            
            self.update_group_allocations()
            
        except Exception as e:
            logging.error(f"Error updating group data: {str(e)}")
    
    def update_group_allocations(self):
        total_value = self.broker.getvalue()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}
        
        for data in self.datas:
            position = self.getposition(data)
            if position.size > 0:
                symbol = data._name
                group = self.asset_groups.get(symbol)
                
                if group is not None:
                    position_value = position.size * data.close[0]
                    self.group_allocations[group] += position_value / total_value
    
    def evaluate_sell_conditions(self, data, current_date):
        symbol = data._name
        position = self.getposition(data)
        
        if position.size <= 0:
            return
            
        current_price = data.close[0]
        entry_price = self.entry_prices.get(data, current_price)
        entry_date = self.position_dates.get(data, current_date)
        
        days_held = (current_date - entry_date).days
        
        profit_pct = (current_price / entry_price - 1) * 100
        
        atr = self.inds[data]['atr'][0]
        
        self.update_trailing_stop(data, current_price, atr)
        
        trailing_stop = self.trailing_stops.get(data)
        stop_triggered = current_price < trailing_stop if trailing_stop else False
        
        take_profit_level = entry_price * (1 + self.p.take_profit_percent / 100.0)
        take_profit_triggered = current_price >= take_profit_level
        
        max_hold_triggered = days_held >= self.p.position_timeout
        
        min_return = days_held * self.p.min_daily_return
        poor_performance = days_held > 5 and profit_pct < min_return
        
        should_sell = stop_triggered or take_profit_triggered or max_hold_triggered or poor_performance
        
        if should_sell:
            reason = "Unknown"
            if stop_triggered:
                reason = "Trailing Stop"
            elif take_profit_triggered:
                reason = "Take Profit"
            elif max_hold_triggered:
                reason = "Max Hold Time"
            elif poor_performance:
                reason = "Poor Performance"
                
            logging.info(f"SELL {symbol}: Price={current_price:.2f}, Entry={entry_price:.2f}, "
                         f"Profit={profit_pct:.2f}%, Days={days_held}, Reason={reason}")
                         
            self.close(data=data)
            
            is_loss = current_price < entry_price
            update_trade_result(symbol, is_loss, current_price, current_date)
    
    def update_trailing_stop(self, data, current_price, atr):
        if data not in self.trailing_stops:
            return
            
        entry_price = self.entry_prices.get(data)
        current_stop = self.trailing_stops[data]
        
        new_stop = current_price - (atr * self.p.trailing_stop_atr_multiple)
        
        if new_stop > current_stop:
            self.trailing_stops[data] = new_stop
            
            if new_stop > entry_price:
                logging.info(f"Raising stop for {data._name} to {new_stop:.2f} (in profit)")
    
    def close_all_positions_for_lockup(self):
        """Close all existing positions to prepare for the lockup period."""
        positions_closed = False
        
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.close(data=data)
                logging.info(f"Closing position in {data._name} for lockup period")
                positions_closed = True
        
        if positions_closed:
            self.open_positions = 0
            self.entry_prices = {}
            self.position_dates = {}
            self.trailing_stops = {}
            logging.info("All positions closed for lockup period")
        else:
            logging.info("No positions to close for lockup period")
    


    def save_best_buy_signals(self, buy_candidates):
        """Save the best buy signals to a parquet file for the live trader."""
        current_date = self.datetime.date()

        # Add data freshness check
        try:
            market_last_date = get_last_trading_date()
            if current_date < market_last_date:
                data_age = (market_last_date - current_date).days
                if data_age > 1:  # More than 1 day old
                    logging.warning(f"⚠️ Data is outdated by {data_age} days. Current: {current_date}, Market: {market_last_date}")
                    logging.warning("Signals may not be reliable. Consider updating data before trading.")
                    # You could either skip signal generation or just warn and continue
                    # return  # Uncomment to skip
        except Exception as e:
            logging.error(f"Error checking data freshness: {str(e)}")

        next_trading_day = get_next_trading_day(current_date)

        # Create a fresh DataFrame for new signals
        signal_data = []
        for d, size, correlation in buy_candidates[:self.p.max_positions]:
            price = round(d.close[0], 3)

            signal_data.append({
                'Symbol': str(d._name),
                'LastBuySignalDate': pd.Timestamp(next_trading_day),
                'LastBuySignalPrice': float(price),
                'IsCurrentlyBought': False,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.NaT,
                'UpProbability': float(d.UpProbability[0]),
                'LastSellPrice': float('nan'),
                'PositionSize': float('nan')
            })

            logging.info(f"Prepared buy signal for {d._name} at {price} for {next_trading_day}")

        if signal_data:
            # Create completely new DataFrame with just these signals
            new_signals_df = pd.DataFrame(signal_data)

            # Option 1: Complete overwrite - replace ALL signals
            # write_trading_data(new_signals_df)

            # Option 2: Maintain currently bought positions but replace all signals
            df = read_trading_data()

            # Keep only currently bought positions
            currently_bought = df[df['IsCurrentlyBought'] == True]

            # Create final DataFrame with both bought positions and new signals
            final_df = pd.concat([currently_bought, new_signals_df], ignore_index=True)

            # Remove duplicates in case a symbol is both bought and in new signals
            final_df = final_df.drop_duplicates(subset=['Symbol'], keep='first')

            write_trading_data(final_df)

            # Synchronize with live trading data
            sync_trading_data()

            logging.info(f"Successfully wrote {len(signal_data)} new buy signals and synchronized with live trader")





    def notify_order(self, order):

        if order.status in [order.Completed, order.Partial]:
            self.handle_order_execution(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.handle_order_failure(order)
    
    def handle_order_execution(self, order):

        if order.isbuy():
            self.handle_buy_execution(order)
        elif order.issell():
            self.handle_sell_execution(order)
    
    def handle_buy_execution(self, order):

        data = order.data
        symbol = data._name

        mark_position_as_bought(symbol, order.executed.size)
        logging.info(f"BUY EXECUTED for {symbol}: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size}, Cost={order.executed.value:.2f}")
    
    def handle_sell_execution(self, order):

        data = order.data
        symbol = data._name
        
        if self.getposition(data).size == 0:
            self.open_positions -= 1
            
            entry_price = self.entry_prices.get(data)
            exit_price = order.executed.price
            entry_date = self.position_dates.get(data)
            exit_date = self.datetime.date()
            
            if entry_price is not None:
                profit_pct = ((exit_price / entry_price) - 1) * 100
                profit_abs = (exit_price - entry_price) * abs(order.executed.size)
                days_held = (exit_date - entry_date).days if entry_date else 0
                
                is_win = profit_abs > 0
                is_loss = profit_abs < 0
                
                if is_win:
                    self.winning_trades += 1
                    self.total_win_pnl += profit_abs
                    self.current_win_streak += 1
                    self.current_loss_streak = 0
                    self.recent_outcomes.append(1)
                    if self.current_win_streak > self.longest_win_streak:
                        self.longest_win_streak = self.current_win_streak
                elif is_loss:
                    self.losing_trades += 1
                    self.total_loss_pnl += profit_abs  # This will be negative
                    self.current_loss_streak += 1
                    self.current_win_streak = 0
                    self.recent_outcomes.append(-1)
                    if self.current_loss_streak > self.longest_loss_streak:
                        self.longest_loss_streak = self.current_loss_streak
                else:
                    self.breakeven_trades += 1
                    self.recent_outcomes.append(0)
                
                if len(self.recent_outcomes) > 10:
                    self.recent_outcomes = self.recent_outcomes[-10:]
                
                trade_data = {
                    'Symbol': symbol,
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Quantity': abs(order.executed.size),
                    'PnL': profit_abs,
                    'PnLPct': profit_pct,
                    'DaysHeld': days_held,
                    'Commission': order.executed.comm,
                    'TradeType': 'Long',
                    'ExitReason': self.determine_exit_reason(data),
                    'ATR': self.inds[data]['atr'][0],
                    'UpProbability': data.UpProbability[0],
                    'AccountValue': self.broker.getvalue(),
                }
                
                self.trade_history.append(trade_data)
                self.trade_recorder.record_trade(trade_data)
                
                is_loss_for_tracking = profit_abs < 0
                update_trade_result(symbol, is_loss_for_tracking, exit_price, exit_date)
            
            if data in self.entry_prices:
                del self.entry_prices[data]
            if data in self.position_dates:
                del self.position_dates[data]
            if data in self.trailing_stops:
                del self.trailing_stops[data]
            if symbol in self.asset_groups:
                del self.asset_groups[symbol]
            
            self.update_group_allocations()
            
            profit_pct = ((exit_price / entry_price) - 1) * 100 if entry_price else 0
            logging.info(f"SELL EXECUTED for {symbol}: Price={exit_price:.2f}, "
                        f"Profit={profit_pct:.2f}%, Value={order.executed.value:.2f}")
    



    def determine_exit_reason(self, data):
        """Determine the reason for exiting a position."""
        symbol = data._name
        current_price = data.close[0]
        entry_price = self.entry_prices.get(data, current_price)
        entry_date = self.position_dates.get(data, self.datetime.date())
        trailing_stop = self.trailing_stops.get(data)
        
        # Calculate metrics
        days_held = (self.datetime.date() - entry_date).days
        profit_pct = (current_price / entry_price - 1) * 100
        take_profit_level = entry_price * (1 + self.p.take_profit_percent / 100.0)
        
        # Check conditions
        if trailing_stop and current_price <= trailing_stop:
            if profit_pct >= 0:
                return "Trailing Stop (In Profit)"
            else:
                return "Stop Loss"
        elif current_price >= take_profit_level:
            return "Take Profit"
        elif days_held >= self.p.position_timeout:
            return "Max Hold Time"
        elif days_held > 5 and profit_pct < (days_held * self.p.min_daily_return):
            return "Poor Performance"
        else:
            return "Manual Exit"
    


    
    def handle_order_failure(self, order):

        if order in self.order_list:
            self.order_list.remove(order)
            
        reason = "Unknown"
        if order.status == order.Canceled:
            reason = "Canceled"
        elif order.status == order.Margin:
            reason = "Insufficient Margin"
        elif order.status == order.Rejected:
            reason = "Rejected"
        elif order.status == order.Expired:
            reason = "Expired"
            
        logging.warning(f"Order failed for {order.data._name}: {reason}")
    
    def stop(self):
        self.progress_bar.close()
        self.trade_recorder.save_trades()
        

##===========================================================[Control]=========================================================##















##===========================================================[Control]=========================================================##













##===========================================================[Control]=========================================================##












##===========================================================[Control]=========================================================##


def run_strategy_optimization(args, logger, param_ranges):
    """Run backtrader optimization to find the best parameters."""
    logger.info("Starting strategy optimization mode")
    
    # Define which parameters to optimize
    optimize_params = args.optimize_param or ['up_prob_threshold', 'stop_loss_atr_multiple', 'trailing_stop_atr_multiple']
    logger.info(f"Parameters being optimized: {optimize_params}")
    
    # Create a parameter space for optimization
    param_ranges = {
        'up_prob_threshold': [0.55, 0.60, 0.65, 0.70, 0.75],
        'up_prob_min_trigger': [0.65, 0.70, 0.75, 0.80, 0.85],
        'max_positions': [2, 3, 4, 5, 6],
        'risk_per_trade_pct': [1.0, 2.0, 3.0, 4.0, 5.0],
        'max_position_pct': [10.0, 15.0, 20.0, 25.0],
        'min_position_pct': [3.0, 5.0, 7.0, 10.0],
        'stop_loss_atr_multiple': [0.5, 0.75, 1.0, 1.25, 1.5],
        'trailing_stop_atr_multiple': [1.0, 1.5, 2.0, 2.5, 3.0],
        'take_profit_percent': [10.0, 15.0, 20.0, 25.0, 30.0],
        'position_timeout': [3, 4, 5, 6, 7],
        'min_daily_return': [0.5, 0.75, 1.0, 1.25, 1.5]
    }
    
    # Create optimization dictionary containing only the parameters to optimize
    optim_dict = {}
    for param in optimize_params:
        if param in param_ranges:
            optim_dict[param] = param_ranges[param]
        else:
            logger.warning(f"Parameter {param} not found in available parameter ranges")
    
    if not optim_dict:
        logger.error("No valid parameters to optimize")
        return None
    
    # Setup Cerebro for optimization
    cerebro = bt.Cerebro(maxcpus=None, optreturn=True)
    cerebro.broker.set_cash(10000)  # Initial cash
    comminfo = FixedCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    
    # Get data files (use a smaller sample for optimization to make it faster)
    data_dir = 'Data/RFpredictions'
    sample_pct = min(args.sample, 25)  # Limit to max 25% for optimization to keep it manageable
    file_paths = select_data_files(args, data_dir, logger)
    
    if not file_paths:
        logger.error("No data files available for optimization")
        return None
    
    # Process data files
    aligned_data = process_data_files(args, file_paths, logger)
    
    if not aligned_data:
        logger.error("No aligned data available for optimization")
        return None
    
    # Add data feeds to cerebro
    for name, df in aligned_data:
        data_feed = EnhancedPandasData(dataname=df)
        cerebro.adddata(data_feed, name=name)
    
    # Add analyzers for optimization metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # Add strategy with optimization parameters
    cerebro.optstrategy(StockSniperStrategy, **optim_dict)
    
    # Run optimization
    logger.info(f"Running optimization with {args.runs} potential combinations")
    try:
        results = cerebro.run()
        
        # Process optimization results
        all_results = []
        for run_params, strategies in results:
            strategy = strategies[0]
            
            # Extract metrics
            sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            returns = strategy.analyzers.returns.get_analysis().get('rtot', 0) * 100  # Convert to percentage
            max_dd = strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            trades = strategy.analyzers.trades.get_analysis()
            sqn = strategy.analyzers.sqn.get_analysis().get('sqn', 0)
            
            # Calculate profit factor
            won_total = trades.get('won', {}).get('pnl', {}).get('total', 0)
            lost_total = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            profit_factor = won_total / lost_total if lost_total > 0 else float('inf')
            
            # Calculate win rate
            total_trades = trades.get('total', {}).get('closed', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate composite score (adjust weights as needed)
            composite_score = (
                sharpe * 0.25 +
                returns * 0.2 +
                sqn * 0.15 +
                win_rate * 0.1 +
                profit_factor * 0.2 -
                max_dd * 0.1
            )
            
            # Create result entry
            result_entry = {
                'params': run_params,
                'sharpe': sharpe,
                'returns': returns,
                'max_dd': max_dd,
                'sqn': sqn,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trades': total_trades,
                'composite_score': composite_score
            }
            
            all_results.append(result_entry)
        
        # Sort by composite score (descending)
        all_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Display top results
        logger.info(f"Optimization completed with {len(all_results)} results")
        logger.info("Top 5 parameter combinations:")
        
        print("\n" + "=" * 80)
        print(" Optimization Results - Top Parameter Combinations ".center(80))
        print("=" * 80 + "\n")
        
        for i, result in enumerate(all_results[:5]):
            print(f"\nRank {i+1}. Composite Score: {result['composite_score']:.2f}")
            print(f"Parameters:")
            for param, value in result['params']._asdict().items():
                print(f"  - {param}: {value}")
            print(f"Performance Metrics:")
            print(f"  - Returns: {result['returns']:.2f}%")
            print(f"  - Sharpe Ratio: {result['sharpe']:.2f}")
            print(f"  - SQN: {result['sqn']:.2f}")
            print(f"  - Win Rate: {result['win_rate']:.2f}%")
            print(f"  - Profit Factor: {result['profit_factor']:.2f}")
            print(f"  - Max Drawdown: {result['max_dd']:.2f}%")
            print(f"  - Total Trades: {result['trades']}")
        
        # Get the best parameters
        best_params = all_results[0]['params']._asdict() if all_results else {}
        
        # Return the best parameter set
        return best_params
    
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.error(traceback.format_exc())
        return None





def generate_best_buy_signals(num_signals=5, min_up_prob=0.65, data_dir='Data/RFpredictions'):
    """
    Generate best buy signals for current or last trading day.
    
    Args:
        num_signals: Number of best signals to generate
        min_up_prob: Minimum UpProbability threshold
        data_dir: Directory containing stock prediction parquet files
        
    Returns:
        List of best stocks with their qualities
    """
    import glob
    import pandas as pd
    import numpy as np
    import os
    from tqdm import tqdm
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_signals} best buy signals")
    
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    logger.info(f"Found {len(all_files)} stock prediction files")
    
    # Get the last trading date
    try:
        last_trading_date = get_last_trading_date()
        logger.info(f"Last trading date: {last_trading_date}")
    except Exception as e:
        logger.warning(f"Error getting last trading date: {e}. Using current date.")
        last_trading_date = datetime.now().date()
    
    # For storing quality scores of stocks
    quality_stocks = []
    
    # Process each stock file
    for file_path in tqdm(all_files, desc="Evaluating stocks"):
        try:
            # Read only the necessary columns for speed
            df = pd.read_parquet(file_path, columns=['Date', 'Open', 'Close', 'UpProbability'])
            
            if len(df) < 5:  # Need at least 5 days of data
                continue
                
            # Convert Date to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date and get most recent data
            df = df.sort_values('Date')
            
            # Get the most recent date in this dataset
            latest_date = df['Date'].max().date()
            
            # If data is too old, skip it
            if (last_trading_date - latest_date).days > 5:
                continue
                
            # Get the latest UpProbability
            latest_prob = df['UpProbability'].iloc[-1]
            
            if latest_prob < min_up_prob:
                continue
                
            # Calculate key metrics for quality score
            # 1. Variance of UpProbability (higher is better)
            up_prob_variance = df['UpProbability'].tail(10).var()
            
            # 2. Trend of UpProbability (positive is better)
            x = np.arange(5)
            y = df['UpProbability'].tail(5).values
            up_prob_trend = np.polyfit(x, y, 1)[0] if len(y) == 5 else 0
            
            # 3. Latest value relative to recent average (higher is better)
            up_prob_ratio = latest_prob / df['UpProbability'].tail(5).mean() if df['UpProbability'].tail(5).mean() > 0 else 1
            
            # 4. Count high probability days (more is better)
            high_days = sum(1 for p in df['UpProbability'].tail(10) if p > 0.6)
            
            # 5. Price momentum (positive is better)
            price_momentum = 0
            if len(df) >= 5:
                recent_prices = df['Close'].tail(5).values
                price_momentum = (recent_prices[-1] / recent_prices[0] - 1) * 100  # Percentage

            # Calculate quality score (higher is better)
            # Weight the components based on importance
            quality_score = (
                latest_prob * 50 +                       # Current probability (0-50 points)
                min(up_prob_variance * 1000, 15) +       # Variance (0-15 points)
                (up_prob_trend * 100) * 10 +             # Trend direction (-10 to +10 points)
                min((up_prob_ratio - 1) * 10, 10) +      # Ratio to average (0-10 points)
                high_days * 0.5 +                        # High days (0-5 points)
                min(max(price_momentum, 0), 10)          # Price momentum (0-10 points)
            )
            
            # Get stock name from file path
            stock_name = os.path.basename(file_path).replace('.parquet', '')
            
            # Add to quality stocks list
            quality_stocks.append({
                'Symbol': stock_name,
                'UpProbability': latest_prob,
                'Quality': quality_score,
                'Variance': up_prob_variance,
                'Trend': up_prob_trend,
                'HighDays': high_days,
                'Price': df['Close'].iloc[-1],
                'Date': latest_date
            })
                
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
    
    # Sort by quality score (highest first)
    quality_stocks.sort(key=lambda x: x['Quality'], reverse=True)
    
    # Take the top N signals
    best_signals = quality_stocks[:num_signals]
    
    if best_signals:
        logger.info("Top quality stocks found:")
        for i, stock in enumerate(best_signals):
            logger.info(f"{i+1}. {stock['Symbol']}: UpProb={stock['UpProbability']:.4f}, Quality={stock['Quality']:.2f}")
    else:
        logger.warning("No quality stocks found matching criteria")
    
    return best_signals



def save_best_signals_to_parquet(signals, next_trading_day=None):
    """
    Save the generated best signals to the buy signals parquet file.
    
    Args:
        signals: List of dictionaries containing signal information
        next_trading_day: Optional next trading day, if None will calculate it
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not signals:
        logger.warning("No signals to save")
        return
    
    try:
        # Get next trading day if not provided
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f"Next trading day: {next_trading_day}")
        
        # Read existing signal file
        try:
            df = read_trading_data()
            logger.info(f"Read existing trading data with {len(df)} records")
        except:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame(columns=[
                'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought',
                'ConsecutiveLosses', 'LastTradedDate', 'UpProbability', 'LastSellPrice', 'PositionSize'
            ])
            logger.info("Created new trading data DataFrame")
        
        # Create new signal data
        signal_data = []
        for signal in signals:
            signal_data.append({
                'Symbol': signal['Symbol'],
                'LastBuySignalDate': pd.Timestamp(next_trading_day),
                'LastBuySignalPrice': float(signal['Price']),
                'IsCurrentlyBought': False,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.NaT,
                'UpProbability': float(signal['UpProbability']),
                'LastSellPrice': float('nan'),
                'PositionSize': float('nan')
            })
        
        # Create new signals DataFrame
        new_signals_df = pd.DataFrame(signal_data)
        
        # Keep only currently bought positions
        currently_bought = df[df['IsCurrentlyBought'] == True]
        
        # Create final DataFrame with both bought positions and new signals
        final_df = pd.concat([currently_bought, new_signals_df], ignore_index=True)
        
        # Remove duplicates in case a symbol is both bought and in new signals
        final_df = final_df.drop_duplicates(subset=['Symbol'], keep='first')
        
        # Write to file
        write_trading_data(final_df)
        
        # Synchronize with live trading data
        sync_trading_data()
        
        logger.info(f"Successfully wrote {len(signal_data)} new buy signals")
        
    except Exception as e:
        logger.error(f"Error saving best signals to parquet: {e}")
        logger.error(traceback.format_exc())







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





def arg_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Sniper Trading Strategy")
    parser.add_argument("--sample", type=float, default=100, help="Percentage of random files to backtest (0-100)")
    parser.add_argument("--filter", type=float, default=0.01,help="Minimum UpProbability variance for stock filtering")
    parser.add_argument("--up_prob", type=float, default=0.68,help="UpProbability threshold for buy signals")
    parser.add_argument("--force", action='store_true', help="Force the script to run even if data is not up to last trading date")
    parser.add_argument("--recommend", action='store_true', default=False, help="Recommend basic system changes based on the backtest risk metrics")
    parser.add_argument("--best", action='store_true', default=False, help="Generate best buy signals for the current or last trading day")
    parser.add_argument("--num_signals", type=int, default=4, help="Number of best signals to generate (default: 5)")
    
    # Add new optimization related arguments
    parser.add_argument("--optimize", action='store_true', default=False, help="Run in optimization mode to find best parameters")
    parser.add_argument("--optimize_param", type=str, action='append', default=None, 
                       help="Parameters to optimize (can be used multiple times, e.g. --optimize_param up_prob_threshold --optimize_param max_positions)")
    parser.add_argument("--runs", type=int, default=10, help="Number of optimization runs (default: 10)")
    
    # Add individual parameter arguments for more granular control
    parser.add_argument("--max_positions", type=int, help="Maximum number of concurrent positions")
    parser.add_argument("--risk_per_trade", type=float, help="Risk per trade percentage")
    parser.add_argument("--stop_loss_atr", type=float, help="Stop loss ATR multiple")
    parser.add_argument("--trailing_stop_atr", type=float, help="Trailing stop ATR multiple")
    parser.add_argument("--take_profit", type=float, help="Take profit percentage")
    parser.add_argument("--position_timeout", type=int, help="Maximum days to hold a position")
    return parser.parse_args()







def filter_stocks_by_signal_quality(data_dir, min_variance=0.1, min_up_prob=0.75):

    import os
    import glob
    import pandas as pd
    import concurrent.futures
    from tqdm import tqdm
    import logging

    
    logger = logging.getLogger(__name__)
    filtered_files = []
    all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
    logger.info(f"Found {len(all_files)} stock prediction files")
    
    def meets_criteria(file_path):
        try:
            df = pd.read_parquet(file_path, columns=['up_prob'])
            
            max_up_prob = df['up_prob'].max()
            if max_up_prob < min_up_prob:
                return (False, f"max_up_prob {max_up_prob:.2f} < {min_up_prob:.2f}")
                
            variance = df['up_prob'].var()
            if variance < min_variance:
                return (False, f"variance {variance:.4f} < {min_variance:.4f}")
                
            return (True, f"Passed: max_up_prob={max_up_prob:.2f}, var={variance:.4f}")
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(meets_criteria, all_files),
            total=len(all_files),
            desc="Filtering stocks by signal quality"
        ))
    
    filtered_files = []
    rejected_counts = {"max_up_prob": 0, "variance": 0, "error": 0}
    
    for file_path, (meets, reason) in zip(all_files, results):
        file_name = os.path.basename(file_path)
        if meets:
            filtered_files.append(file_path)
            logger.debug(f"Accepted {file_name}: {reason}")
        else:
            if "max_up_prob" in reason:
                rejected_counts["max_up_prob"] += 1
            elif "variance" in reason:
                rejected_counts["variance"] += 1
            else:
                rejected_counts["error"] += 1
            logger.debug(f"Rejected {file_name}: {reason}")
    
    logger.info(f"Filtered to {len(filtered_files)} stocks with quality signals")
    logger.info(f"Rejected: {rejected_counts['max_up_prob']} for low up_prob, " 
                f"{rejected_counts['variance']} for low variance, "
                f"{rejected_counts['error']} due to errors")
    
    return filtered_files




# ------------------------------------------------------------------------------
# Main function and setup routines
# ------------------------------------------------------------------------------

def main():
    """Main function for running the StockSniper Strategy backtest."""
    logger = get_logger(script_name="5__NightlyBackTester")
    start_time = time.time()
    
    try:
        # Setup phase
        args = arg_parser()

        if args.best:
            logger.info("Best signals generation mode activated")
            best_signals = generate_best_buy_signals(num_signals=args.num_signals, 
                                                    min_up_prob=args.up_prob)
                                                    
            if best_signals:
                save_best_signals_to_parquet(best_signals)
                logger.info(f"Successfully generated {len(best_signals)} best buy signals")
                return {"status": "success", "signals": len(best_signals)}
            else:
                logger.error("Failed to generate best buy signals")
                return {"status": "error", "message": "No signals generated"}


        cerebro, data_feeds = setup_backtest_environment(args, logger)
        
        if not data_feeds:
            logger.error("No data feeds available. Exiting.")
            return None
            
        # Run backtest
        strategies = cerebro.run()
        if not strategies:
            logger.error("No strategies were executed.")
            return None
            
        # Process results
        first_strategy = strategies[0]
        results = extract_backtest_results(first_strategy, cerebro, logger)
        
        # Compute execution time
        execution_time = time.time() - start_time
        
        # Display results
        print_detailed_results(results, execution_time)
        #log_summary_results(results, data_feeds, logger)
        
        # Optional plotting
        try_plot_results(cerebro, logger)
        
        # Check for buy signals
        signals_verified = check_buy_signals(logger)
        
        # Run post-backtest organization
        org_success = run_post_backtest_organization()
        
        # Verify the entire handoff process if both steps were successful
        if signals_verified and org_success:
            logger.info("Signal generation and organization completed successfully")
        else:
            logger.warning("Issues detected in signal generation or organization")
        
        # Return summary
        return create_results_summary(results)
    
    except Exception as e:
        logger.error(f"Critical error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\nA critical error occurred: {str(e)}")
        print("Check the log file for details.")
        return None


def setup_backtest_environment(args, logger):
    """Set up the backtest environment with Cerebro and data feeds."""
    cerebro = bt.Cerebro(maxcpus=None)
    cerebro.broker.set_cash(10000)  # Initial cash
    comminfo = FixedCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.set_coc(False)  # Close position at end of day
    
    # Get data files
    data_dir = 'Data/RFpredictions'
    file_paths = select_data_files(args, data_dir, logger)
    
    if not file_paths:
        return cerebro, []
    
    # Process data files
    aligned_data = process_data_files(args, file_paths, logger)
    
    if not aligned_data:
        return cerebro, []
    
    # Add data feeds to cerebro
    for name, df in aligned_data:
        data_feed = EnhancedPandasData(dataname=df)
        cerebro.adddata(data_feed, name=name)
    
    # Add analyzers
    add_analyzers(cerebro, logger)
    
    # Add strategy with parameters from command line arguments or use defaults from STRATEGY_PARAMS
    strategy_params = {}
    
    # Override parameters with command line arguments if provided
    if hasattr(args, 'up_prob') and args.up_prob is not None:
        strategy_params['up_prob_threshold'] = args.up_prob
        strategy_params['up_prob_min_trigger'] = args.up_prob + 0.02  # Slightly higher for strongest signals
    
    # Add any other parameter overrides from command line arguments here
    # For example:
    # if hasattr(args, 'max_positions') and args.max_positions is not None:
    #     strategy_params['max_positions'] = args.max_positions
    
    # Add the strategy with the combined parameters
    cerebro.addstrategy(StockSniperStrategy, **strategy_params)
    
    return cerebro, aligned_data












def select_data_files(args, data_dir, logger):
    """Select data files based on sampling or filtering criteria."""
    if args.sample > 0:
        all_files = glob.glob(os.path.join(data_dir, '*.parquet'))
        num_files = len(all_files)
        num_to_select = max(1, int(round(num_files * args.sample / 100)))
        file_paths = random.sample(all_files, num_to_select)
        logger.info(f"Selected {len(file_paths)} random files ({args.sample}% of {num_files})")
    else:
        file_paths = filter_stocks_by_signal_quality(
            data_dir, 
            min_variance=args.filter,
            min_up_prob=args.up_prob
        )
        
    if not file_paths:
        logger.error("No stock files found or passed filtering. Exiting.")
        return []
        
    logger.info(f"Processing {len(file_paths)} stock files")
    return file_paths


def process_data_files(args, file_paths, logger):
    """Load and process data files, ensuring proper date alignment."""
    last_trading_date = get_last_trading_date()
    logger.info(f"Last trading date: {last_trading_date}")
    
    loaded_data = parallel_load_data(file_paths, last_trading_date)
    
    if args.force:
        max_dates = [df['Date'].dt.date.max() for _, df in loaded_data]
        date_counts = Counter(max_dates)
        if date_counts:
            last_trading_date, count = date_counts.most_common(1)[0]
            logger.info(f"Force mode: Using last trading date: {last_trading_date}")
    
    aligned_data = []
    for name, df in loaded_data:
        max_date = df['Date'].dt.date.max()
        if max_date == last_trading_date and len(df) >= 252:
            aligned_data.append((name, df))
    
    if not aligned_data:
        logger.error("No data remains after alignment. Exiting.")
        return []
    
    logger.info(f"Final dataset: {len(aligned_data)} stocks with {len(aligned_data[0][1])} trading days")
    return aligned_data


def add_analyzers(cerebro, logger):
    """Add analyzers to the Cerebro instance."""
    analyzers_to_add = [
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeStats"}),
        (bt.analyzers.DrawDown, {"_name": "DrawDown"}),
        (bt.analyzers.SharpeRatio, {"_name": "SharpeRatio", "riskfreerate": 0.05}),
        (bt.analyzers.SQN, {"_name": "SQN"}),
        (bt.analyzers.Returns, {"_name": "Returns"}),
        (bt.analyzers.VWR, {"_name": "VWR"}),
        (bt.analyzers.TimeReturn, {"_name": "TimeReturn"}),
        (bt.analyzers.PeriodStats, {"_name": "PeriodStats"}),
        (bt.analyzers.Transactions, {"_name": "Transactions"}),
        (bt.analyzers.TradeAnalyzer, {"_name": "TradeAnalyzer"}),
        (bt.analyzers.PositionsValue, {"_name": "PositionsValue"}),
        (bt.analyzers.TimeDrawDown, {"_name": "TimeDrawDown"}),
        (bt.analyzers.PyFolio, {"_name": "PyFolio"})
    ]
    
    for analyzer_class, kwargs in analyzers_to_add:
        try:
            cerebro.addanalyzer(analyzer_class, **kwargs)
            logger.debug(f"Added analyzer: {kwargs.get('_name', 'unnamed')}")
        except Exception as e:
            logger.error(f"Failed to add analyzer {kwargs.get('_name', 'unnamed')}: {str(e)}")


# ------------------------------------------------------------------------------
# Results extraction and processing routines
# ------------------------------------------------------------------------------

def extract_backtest_results(strategy, cerebro, logger):
    """Extract detailed results from the backtest."""
    results = initialize_results_dict(cerebro)
    
    try:
        # Set day count
        try:
            day_count = strategy.day_count
        except AttributeError:
            logger.warning("Could not get day_count from strategy, using 252 as fallback")
            day_count = 252
            
        results['day_count'] = day_count
        
        # Calculate total and annualized returns
        results['total_return'] = (results['final_value'] / results['initial_value'] - 1) * 100
        try:
            results['annualized_return'] = ((results['final_value'] / results['initial_value']) ** (252 / day_count) - 1) * 100
        except Exception as e:
            logger.warning(f"Failed to calculate annualized return: {str(e)}")
        
        # Extract data from analyzers
        analyzer_data = get_analyzer_data(strategy, logger)
        
        # Get strategy-specific data
        if hasattr(strategy, 'monthly_performance'):
            results['monthly_performance'] = strategy.monthly_performance
        
        if hasattr(strategy, 'yearly_performance'):
            results['yearly_performance'] = strategy.yearly_performance
        
        # Process analyzer data into results
        process_trade_statistics(results, analyzer_data, logger)
        process_drawdown_statistics(results, analyzer_data, logger)
        process_sharpe_ratio(results, analyzer_data, logger)
        process_trade_analyzer_data(results, analyzer_data, logger)
        process_daily_returns_data(results, analyzer_data, logger)
        calculate_risk_of_ruin(results, logger)
        determine_sqn_description(results)
    
        try:
            # Add enhanced metrics to results dictionary
            results = integrate_enhanced_metrics_into_results(results)
            logger.info(f"Enhanced metrics calculated. Modified SQN: {results.get('enhanced_modified_sqn', 0):.2f}")
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {str(e)}")
            logger.error(traceback.format_exc())

        return results


    except Exception as e:
        logger.error(f"Error extracting metrics from analyzers: {str(e)}")
    
    return results


def initialize_results_dict(cerebro):
    """Initialize the results dictionary with default values."""
    return {
        # Core metrics
        'initial_value': cerebro.broker.startingcash,
        'final_value': cerebro.broker.getvalue(),
        'total_return': 0,
        'annualized_return': 0,
        'daily_return': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'calmar_ratio': 0,
        'sqn_value': 0,
        'sqn_description': "Unknown",
        'vwr': 0,
        'gain_to_pain_ratio': 0,
        'omega_ratio': 0,
        'information_ratio': 0,
        
        # Risk metrics
        'max_dd': 0,
        'max_dd_duration': 0,
        'avg_dd': 0,
        'avg_dd_duration': 0,
        'ulcer_index': 0,
        'recovery_factor': 0,
        'common_sense_ratio': 0,
        'risk_of_ruin': 1.0,
        'daily_volatility': 0,
        'annualized_volatility': 0,
        'var_95': 0,
        'cvar_95': 0,
        
        # Trade statistics
        'total_closed': 0,
        'won_total': 0,
        'lost_total': 0,
        'won_pnl_total': 0,
        'lost_pnl_total': 0,
        'won_avg': 0,
        'lost_avg': 0,
        'won_max': 0,
        'lost_max': 0,
        'net_total': 0,
        'profit_factor': 0,
        'percent_profitable': 0,
        'risk_reward_ratio': 0,
        'expectancy': 0,
        'kelly_percentage': 0,
        'avg_win_pct': 0,
        'avg_loss_pct': 0,
        'largest_win_pct': 0,
        'largest_loss_pct': 0,
        'avg_profit_per_trade': 0,
        'net_profit_drawdown_ratio': 0,
        
        # Trade management
        'avg_trade_len': 0,
        'longest_trade': 0,
        'shortest_trade': 0,
        'time_in_market_pct': 0,
        'max_consecutive_wins': 0,
        'max_consecutive_losses': 0,
        'current_streak': None,
        'win_loss_count_ratio': 0,
        
        # Advanced metrics
        'positive_days_pct': 0.0,
        'max_pos_streak': 0,
        'max_neg_streak': 0,
        'mfe_avg': 0,
        'mae_avg': 0,
        'mfe_max': 0,
        'mae_max': 0,
        'profit_per_day': 0,
        
        # Strategy specific
        'monthly_performance': {},
        'yearly_performance': {}
    }


def get_analyzer_data(strategy, logger):
    """Safely extract data from all analyzers."""
    analyzer_data = {}
    
    # Define the list of analyzers to extract: (key, analyzer_name)
    analyzers = [
        ('trade_stats', 'TradeStats'),
        ('drawdown', 'DrawDown'),
        ('sharpe_ratio', 'SharpeRatio'),
        ('sqn', 'SQN'),
        ('returns', 'Returns'),
        ('vwr', 'VWR'),
        ('time_return', 'TimeReturn'),
        ('period_stats', 'PeriodStats'),
        ('transactions', 'Transactions'),
        ('trade_analyzer', 'TradeAnalyzer'),
        ('positions_value', 'PositionsValue'),
        ('time_drawdown', 'TimeDrawDown')
    ]
    
    # Extract data from each analyzer with error handling
    for key, name in analyzers:
        try:
            analyzer = getattr(strategy.analyzers, name)
            analyzer_data[key] = analyzer.get_analysis()
            
            # Fix for SQN
            if key == 'sqn':
                sqn_value = analyzer_data[key].get('sqn', None)
                if sqn_value is not None:
                    strategy.sqn_value = sqn_value
                    logger.info(f"SQN value: {sqn_value}")
                else:
                    # Try to calculate SQN manually if analyzer doesn't provide it
                    if 'trade_stats' in analyzer_data and strategy.winning_trades + strategy.losing_trades > 0:
                        trade_results = []
                        won_total = analyzer_data['trade_stats'].get('won', {}).get('total', 0)
                        lost_total = analyzer_data['trade_stats'].get('lost', {}).get('total', 0)
                        won_pnl = analyzer_data['trade_stats'].get('won', {}).get('pnl', {}).get('total', 0)
                        lost_pnl = analyzer_data['trade_stats'].get('lost', {}).get('pnl', {}).get('total', 0)
                        
                        total_trades = won_total + lost_total
                        if total_trades > 0:
                            avg_win = won_pnl / won_total if won_total > 0 else 0
                            avg_loss = lost_pnl / lost_total if lost_total > 0 else 0
                            
                            # Approximate trade results for SQN calculation
                            trade_results = [avg_win] * won_total + [avg_loss] * lost_total
                            
                            if trade_results:
                                mean_r = sum(trade_results) / len(trade_results)
                                std_dev = (sum((r - mean_r) ** 2 for r in trade_results) / len(trade_results)) ** 0.5
                                
                                if std_dev > 0:
                                    strategy.sqn_value = (mean_r / std_dev) * (len(trade_results) ** 0.5)
                                    logger.info(f"Manually calculated SQN value: {strategy.sqn_value}")
            elif key == 'returns':
                strategy.daily_return = analyzer_data[key].get('rtot', 0) / strategy.day_count
            elif key == 'vwr':
                strategy.vwr = analyzer_data[key].get('vwr', 0)
            elif key == 'period_stats':
                strategy.time_in_market_pct = analyzer_data[key].get('inmarket', 0) * 100
                
        except Exception as e:
            logger.warning(f"Failed to get {name} analysis: {str(e)}")
            analyzer_data[key] = {}
    
    return analyzer_data

def process_trade_statistics(results, analyzer_data, logger):
    """Process trade statistics from the analyzer data."""
    try:
        trade_stats = analyzer_data['trade_stats']
        
        results['total_closed'] = trade_stats.get('total', {}).get('closed', 0)
        results['won_total'] = trade_stats.get('won', {}).get('total', 0)
        results['lost_total'] = trade_stats.get('lost', {}).get('total', 0)
        
        results['won_pnl_total'] = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
        results['lost_pnl_total'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('total', 0))
        
        results['won_avg'] = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
        results['lost_avg'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('average', 0))
        
        results['won_max'] = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
        results['lost_max'] = abs(trade_stats.get('lost', {}).get('pnl', {}).get('max', 0))
        
        results['net_total'] = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)
        
        # Calculate derived metrics
        if results['lost_pnl_total'] > 0:
            results['profit_factor'] = results['won_pnl_total'] / results['lost_pnl_total']
        else:
            results['profit_factor'] = float('inf')
            
        if results['total_closed'] > 0:
            results['percent_profitable'] = (results['won_total'] / results['total_closed'] * 100)
        
        # Calculate average trade size and commission impact
        if 'avg_trade_len' in results:
            avg_trade_price = results['initial_value'] / 50  # Rough approximation of average position size
            avg_trade_size = avg_trade_price * results['avg_trade_len'] / 252  # Size based on duration
        else:
            avg_trade_price = results['initial_value'] / 50
            avg_trade_size = avg_trade_price
            
        # Calculate commission impact as percentage
        commission_per_trade = 3.0  # Fixed commission from your strategy
        commission_impact_pct = (commission_per_trade / avg_trade_price) * 100 if avg_trade_price > 0 else 0
        
        # Store commission metrics
        results['avg_trade_price'] = avg_trade_price
        results['commission_impact_pct'] = commission_impact_pct
        results['breakeven_threshold_pct'] = commission_impact_pct * 2  # Entry and exit commissions
        
        # Estimate gross win rate (before commissions)
        if results['total_closed'] > 0:
            # Estimate number of marginally profitable trades that become losers due to commission
            marginal_trades = sum(1 for trade in trade_stats.get('trades', []) 
                               if isinstance(trade, dict) and 
                               0 < trade.get('pnl', 0) < commission_per_trade * 2)
            
            # If we can't get individual trades, make an estimation based on average profits
            if marginal_trades == 0 and results['won_avg'] > 0:
                # Estimate percentage of winning trades that would be losers if commissions were higher
                margin_pct = min(1.0, (commission_per_trade * 2) / results['won_avg'])
                marginal_trades = int(results['won_total'] * margin_pct * 0.2)  # Assume 20% are near the threshold
            
            # Calculate gross win rate (adding back marginal trades)
            results['gross_win_rate'] = ((results['won_total'] + marginal_trades) / results['total_closed']) * 100
        else:
            results['gross_win_rate'] = 0.0
            
        if results['lost_avg'] > 0:
            results['risk_reward_ratio'] = abs(results['won_avg'] / results['lost_avg'])
        else:
            results['risk_reward_ratio'] = float('inf')
        
        p_win = results['percent_profitable'] / 100
        results['expectancy'] = (p_win * results['won_avg']) + ((1 - p_win) * -results['lost_avg'])
        
        if results['risk_reward_ratio'] > 0:
            results['kelly_percentage'] = ((p_win) - ((1 - p_win) / results['risk_reward_ratio'])) * 100
        
        if results['total_closed'] > 0:
            results['avg_profit_per_trade'] = results['net_total'] / results['total_closed']
        
        # Percentage metrics
        results['avg_win_pct'] = results['won_avg'] / results['initial_value'] * 100
        results['avg_loss_pct'] = results['lost_avg'] / results['initial_value'] * 100
        results['largest_win_pct'] = results['won_max'] / results['initial_value'] * 100
        results['largest_loss_pct'] = results['lost_max'] / results['initial_value'] * 100
        
        if results['lost_total'] > 0:
            results['win_loss_count_ratio'] = results['won_total'] / results['lost_total']
        else:
            results['win_loss_count_ratio'] = float('inf')
        
        if results['day_count'] > 0:
            results['profit_per_day'] = results['net_total'] / results['day_count']
            
    except Exception as e:
        logger.warning(f"Error processing trade statistics: {str(e)}")




def process_drawdown_statistics(results, analyzer_data, logger):
    """Process drawdown statistics from the analyzer data."""
    try:
        drawdown = analyzer_data['drawdown']
        
        results['max_dd'] = drawdown.get('max', {}).get('drawdown', 0)
        results['max_dd_duration'] = drawdown.get('max', {}).get('len', 0)
        results['avg_dd'] = drawdown.get('average', {}).get('drawdown', 0)
        results['avg_dd_duration'] = drawdown.get('average', {}).get('len', 0)
        
        # Calculate derived metrics
        if results['max_dd'] > 0:
            results['calmar_ratio'] = results['annualized_return'] / results['max_dd']
        else:
            results['calmar_ratio'] = float('inf')
        
        if results['max_dd'] > 0:
            results['recovery_factor'] = results['total_return'] / results['max_dd']
        else:
            results['recovery_factor'] = float('inf')
        
        if results['total_return'] > 0 and results['max_dd'] > 0:
            results['common_sense_ratio'] = results['total_return'] / results['max_dd_duration']
        
        if results['max_dd'] > 0:
            results['net_profit_drawdown_ratio'] = results['net_total'] / (results['max_dd'] * results['initial_value'] / 100)
        else:
            results['net_profit_drawdown_ratio'] = float('inf')
    except Exception as e:
        logger.warning(f"Error processing drawdown statistics: {str(e)}")


def process_sharpe_ratio(results, analyzer_data, logger):
    """Process Sharpe ratio from the analyzer data."""
    try:
        results['sharpe_ratio'] = analyzer_data['sharpe_ratio'].get('sharperatio', 0)
    except Exception as e:
        logger.warning(f"Error processing Sharpe ratio: {str(e)}")


def process_trade_analyzer_data(results, analyzer_data, logger):
    """Process trade analyzer data for streaks and trade lengths."""
    try:
        trade_analyzer = analyzer_data['trade_analyzer']
        streak_data = trade_analyzer.get('streak', {})
        won_streak = streak_data.get('won', {})
        lost_streak = streak_data.get('lost', {})
        
        results['max_consecutive_wins'] = won_streak.get('longest', 0)
        results['max_consecutive_losses'] = lost_streak.get('longest', 0)
        
        if 'current' in streak_data:
            if streak_data['current'] > 0:
                results['current_streak'] = f"{streak_data['current']} wins"
            elif streak_data['current'] < 0:
                results['current_streak'] = f"{abs(streak_data['current'])} losses"
        
        trade_len = trade_analyzer.get('len', {})
        results['avg_trade_len'] = trade_len.get('average', 0)
        results['longest_trade'] = trade_len.get('max', 0)
        results['shortest_trade'] = trade_len.get('min', 0)
        
        mfe_stats = trade_analyzer.get('mfe', {})
        mae_stats = trade_analyzer.get('mae', {})
        
        results['mfe_avg'] = mfe_stats.get('average', 0)
        results['mfe_max'] = mfe_stats.get('max', 0)
        results['mae_avg'] = mae_stats.get('average', 0)
        results['mae_max'] = mae_stats.get('max', 0)
    except Exception as e:
        logger.warning(f"Error processing trade analyzer data: {str(e)}")


def process_daily_returns_data(results, analyzer_data, logger):
    """Process daily returns data for volatility and related metrics."""
    try:
        daily_returns = []
        for date, ret in analyzer_data['time_return'].items():
            if isinstance(ret, (int, float)):
                daily_returns.append(ret)
                
        if not daily_returns:
            return
            
        # Sortino ratio calculation
        downside_returns = [r for r in daily_returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        if downside_deviation > 0:
            results['sortino_ratio'] = (results['annualized_return'] - 5) / downside_deviation
        else:
            results['sortino_ratio'] = float('inf')
        
        # Gain to pain ratio
        sum_of_positive_returns = sum(max(0, r) for r in daily_returns)
        sum_of_negative_returns = abs(sum(min(0, r) for r in daily_returns))
        if sum_of_negative_returns > 0:
            results['gain_to_pain_ratio'] = sum_of_positive_returns / sum_of_negative_returns
        else:
            results['gain_to_pain_ratio'] = float('inf')
        
        # Ulcer index calculation
        equity_curve = [results['initial_value'] * (1 + r / 100) for r in np.cumsum(daily_returns)]
        drawdowns = []
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
                drawdowns.append(0)
            else:
                dd_pct = (peak - value) / peak * 100
                drawdowns.append(dd_pct)
        results['ulcer_index'] = np.sqrt(np.mean(np.array(drawdowns) ** 2))
        
        # Volatility metrics
        results['daily_volatility'] = np.std(daily_returns) * 100
        results['annualized_volatility'] = results['daily_volatility'] * np.sqrt(252)
        
        # VaR and CVaR
        if len(daily_returns) > 5:
            results['var_95'] = np.percentile(daily_returns, 5) * 100
            cvar_values = [r for r in daily_returns if r < results['var_95'] / 100]
            if cvar_values and results['var_95'] < 0:
                results['cvar_95'] = np.mean(cvar_values) * 100
        
        # Omega ratio
        gains = [r for r in daily_returns if r > 0]
        losses = [abs(r) for r in daily_returns if r < 0]
        if sum(losses) > 0:
            results['omega_ratio'] = sum(gains) / sum(losses)
        else:
            results['omega_ratio'] = float('inf')
        
        # Streak and positive days analysis
        if len(daily_returns) > 20:
            results['positive_days_pct'] = sum(1 for r in daily_returns if r > 0) / len(daily_returns) * 100
            
            pos_streak = 0
            max_pos_streak = 0
            neg_streak = 0
            max_neg_streak = 0
            
            for r in daily_returns:
                if r > 0:
                    pos_streak += 1
                    neg_streak = 0
                    max_pos_streak = max(pos_streak, max_pos_streak)
                else:
                    neg_streak += 1
                    pos_streak = 0
                    max_neg_streak = max(neg_streak, max_neg_streak)
                    
            results['max_pos_streak'] = max_pos_streak
            results['max_neg_streak'] = max_neg_streak
    except Exception as e:
        logger.warning(f"Error calculating advanced metrics from daily returns: {str(e)}")


def calculate_risk_of_ruin(results, logger):
    """Calculate risk of ruin based on win rate and risk/reward ratio."""
    try:
        if results['percent_profitable'] > 0 and results['risk_reward_ratio'] > 0:
            win_rate_decimal = results['percent_profitable'] / 100
            edge = win_rate_decimal - (1 - win_rate_decimal) / results['risk_reward_ratio']
            if edge > 0:
                results['risk_of_ruin'] = ((1 - edge) / (1 + edge)) ** 20
            else:
                results['risk_of_ruin'] = 1.0
    except Exception as e:
        logger.warning(f"Error calculating risk of ruin: {str(e)}")


def determine_sqn_description(results):
    """Determine the SQN description based on the SQN value."""
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
        if low <= results['sqn_value'] < high:
            results['sqn_description'] = desc
            break


# ------------------------------------------------------------------------------
# Printing routines for results
# ------------------------------------------------------------------------------

def print_detailed_results(results, execution_time):
    """Print detailed results to the console with colorized output."""
    print("\n" + "=" * 80)
    print(" Stock Sniper Strategy Backtest Results ".center(80))
    print("=" * 80)

    # Core Performance Metrics
    print("\nCore Performance Metrics:")
    print(colorize_output(results['total_return'], "Total Return %:", 50, 10))
    print(colorize_output(results['annualized_return'], "Annualized Return %:", 25, 10))
    print(colorize_output(results['final_value'], "Final Portfolio Value:", results['initial_value'] * 1.5, results['initial_value'] * 1.1))
    print(colorize_output(results['initial_value'], "Initial Portfolio Value:", results['initial_value'], results['initial_value']))
    print(colorize_output(results['sharpe_ratio'], "Sharpe Ratio:", 1.5, 0.75))
    print(colorize_output(results['sortino_ratio'], "Sortino Ratio:", 2.0, 1.0))
    print(colorize_output(results['calmar_ratio'], "Calmar Ratio:", 2.0, 0.5))
    print(colorize_output(results['gain_to_pain_ratio'], "Gain to Pain Ratio:", 1.5, 1.0))
    print(colorize_output(results['omega_ratio'], "Omega Ratio:", 1.5, 1.0))
    
    # SQN Metrics (Original and Enhanced)
    print(colorize_output(results['sqn_value'], "SQN:", 3.0, 1.6))
    print_sqn_quality(results)
    print(colorize_output(results['vwr'], "Variability-Weighted Return:", 5, 0.5))
    
    # Add enhanced SQN if available
    if 'enhanced_modified_sqn' in results:
        print(colorize_output(results['enhanced_modified_sqn'], "Modified SQN (% normalized):", 3.0, 1.6))
        if 'enhanced_modified_sqn_quality' in results:
            sqn_quality = results['enhanced_modified_sqn_quality']
            print(f"{'Modified SQN Quality:':<30}{get_color_for_sqn(sqn_quality)}{sqn_quality:<10}\033[0m")

    # Risk metrics
    print_risk_metrics(results)
    
    # Trade statistics
    print_trade_statistics(results)
    
    # Trade management metrics
    print_trade_management_metrics(results)
    
    # Advanced trade quality metrics
    print_advanced_trade_metrics(results)
    
    # Enhanced strategy consistency metrics (if available)
    #print_enhanced_consistency_metrics(results)
    
    # Position sizing recommendations (if available)
    #print_position_sizing_recommendations(results)
    
    # Monthly and yearly performance
    print_period_performance(results)
    
    # System interpretation (if enhanced metrics available)
    if 'enhanced_system_robustness_score' in results:
        print_system_interpretation(results)
    
    # Execution time and trade data notice
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print("Trade data saved to trade_history.parquet for further analysis")







def print_sqn_quality(results):
    """Print colorized SQN quality description."""
    sqn_value = results['sqn_value']
    sqn_description = results['sqn_description']
    
    if sqn_value is not None and not math.isnan(sqn_value):
        sqn_color_map = {
            "Holy Grail Potential": 0.0,  # Best
            "Superb": 0.1,
            "Excellent": 0.2,
            "Good": 0.3,
            "Average": 0.5,
            "Below Average": 0.7,
            "Poor": 0.85,
            "Negative": 1.0  # Worst
        }
        normalized_value = sqn_color_map.get(sqn_description, 0.5)  # Default to Average

        if sqn_value >= 15:  # Unicorn level for SQN
            color_code = "\033[38;2;100;149;237m"  # Cornflower blue
            sqn_description = "Unicorn"
        else:
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
            r = int(colors[index][0] * (1 - t) + colors[index+1][0] * t)
            g = int(colors[index][1] * (1 - t) + colors[index+1][1] * t)
            b = int(colors[index][2] * (1 - t) + colors[index+1][2] * t)
            color_code = f"\033[38;2;{r};{g};{b}m"
        print(f"{'SQN Quality:':<30}{color_code}{sqn_description:<10}\033[0m")
    else:
        print(f"{'SQN Quality:':<30}\033[38;2;150;150;150mN/A        \033[0m")


def print_risk_metrics(results):
    """Print risk metrics with colorized output."""
    print("\nRisk Metrics:")
    print(colorize_output(results['max_dd'], "Max Drawdown %:", 10, 25, lower_is_better=True))
    print(colorize_output(results['max_dd_duration'], "Max Drawdown Duration (days):", results['max_consecutive_wins'] * 10, results['max_consecutive_wins'] * 15, lower_is_better=True))
    print(colorize_output(results['ulcer_index'], "Ulcer Index:", 1, 3, lower_is_better=True, unicorn_multiplier=10000.0))
    print(colorize_output(results['recovery_factor'], "Recovery Factor:", 3.0, 1.0))
    print(colorize_output(results['common_sense_ratio'], "Common Sense Ratio:", 0.5, 0.2))
    print(colorize_output(results['risk_of_ruin'], "Risk of Ruin:", 0.001, 0.05, lower_is_better=True, unicorn_multiplier=10000.0))
    print(colorize_output(results['daily_volatility'], "Daily Volatility %:", results['avg_loss_pct'] * 0.8, results['avg_win_pct'], lower_is_better=True))
    print(colorize_output(results['annualized_volatility'], "Annualized Volatility %:", results['avg_loss_pct'] * 0.8 * 15.8, results['avg_win_pct'] * 15.8, lower_is_better=True))
    print(colorize_output(results['var_95'], "Daily VaR (95%):", results['avg_loss_pct'] * 1.2, results['avg_win_pct'] * 1.8, lower_is_better=True))
    print(colorize_output(results['cvar_95'], "Daily CVaR (95%):", results['avg_loss_pct'] * 1.5, results['avg_win_pct'] * 2, lower_is_better=True))


def print_trade_statistics(results):
    """Print trade statistics with colorized output."""
    print("\nTrade Statistics:")
    print(colorize_output(results['total_closed'], "Total Trades:", 50, 10))
    print(colorize_output(results['percent_profitable'], "Win Rate (after fees) %:", 60, 40))
    
    if 'gross_win_rate' in results:
        print(colorize_output(results['gross_win_rate'], "Win Rate (before fees) %:", 60, 40))
        print(colorize_output(results['commission_impact_pct'], "Commission Impact %:", 0.5, 2.0, lower_is_better=True))
        print(colorize_output(results['breakeven_threshold_pct'], "Breakeven Threshold %:", 1.0, 4.0, lower_is_better=True))
    
    print(colorize_output(results['won_avg'], "Avg. Winning Trade ($):", 100, 50))
    print(colorize_output(results['lost_avg'], "Avg. Losing Trade ($):", results['won_avg'] * 0.5, results['won_avg'] * 0.75, lower_is_better=True))
    print(colorize_output(results['avg_win_pct'], "Avg. Winning Trade (%):", 1.0, 0.5))
    print(colorize_output(results['avg_loss_pct'], "Avg. Losing Trade (%):", results['avg_win_pct'] * 0.5, results['avg_win_pct'] * 0.75, lower_is_better=True))
    print(colorize_output(results['won_max'], "Largest Win ($):", results['initial_value'] / 4, 200))
    print(colorize_output(results['lost_max'], "Largest Loss ($):", results['won_max'] * 0.25, results['won_max'] * 0.5, lower_is_better=True))
    print(colorize_output(results['largest_win_pct'], "Largest Win (%):", 5.0, 2.0))
    print(colorize_output(results['largest_loss_pct'], "Largest Loss (%):", results['largest_win_pct'] * 0.5, results['largest_win_pct'] * 2.0, lower_is_better=True))
    print(colorize_output(results['avg_profit_per_trade'], "Avg. Trade P&L:", 50, 0))
    print(colorize_output(results['profit_factor'], "Profit Factor:", 3.0, 1.5))
    print(colorize_output(results['net_profit_drawdown_ratio'], "Net Profit / Drawdown Ratio:", 3.0, 1.0))


def print_trade_management_metrics(results):
    """Print trade management metrics with colorized output."""
    print("\nTrade Management Metrics:")
    print(colorize_output(results['avg_trade_len'], "Avg. Holding Period (days):", 1, 5, lower_is_better=True))
    print(colorize_output(results['longest_trade'], "Longest Trade (days):", 15, 25, lower_is_better=True))
    print(colorize_output(results['shortest_trade'], "Shortest Trade (days):", 1, 5))
    print(colorize_output(results['max_consecutive_wins'], "Max Consecutive Wins:", 5, 3))
    print(colorize_output(results['max_consecutive_losses'], "Max Consecutive Losses:", max(1, results['max_consecutive_wins'] - 1), results['max_consecutive_wins'] + 1, lower_is_better=True))
    print(f"{'Current Streak:':<30}{results['current_streak'] if results['current_streak'] else 'None'}")
    print(colorize_output(results['win_loss_count_ratio'], "Win/Loss Count Ratio:", 1.5, 0.8))
    print(colorize_output(results['risk_reward_ratio'], "Risk/Reward Ratio:", 2.5, 1.0))
    print(colorize_output(results['kelly_percentage'], "Kelly %:", 20, 5))


def print_advanced_trade_metrics(results):
    """Print advanced trade quality metrics with colorized output."""
    print("\nAdvanced Trade Quality Metrics:")
    print(colorize_output(results['positive_days_pct'], "Percentage of Positive Days:", 50, 20))
    print(colorize_output(results['max_pos_streak'], "Max Consecutive Positive Days:", 5, 3))
    print(colorize_output(results['max_neg_streak'], "Max Consecutive Negative Days:", results['max_pos_streak'], results['max_pos_streak'] * 10, lower_is_better=True))
    print(colorize_output(results['profit_per_day'], "Profit per Day ($):", 20, 5))


def print_period_performance(results):
    """Print monthly and yearly performance metrics."""
    if results['monthly_performance']:
        print("\nMonthly Performance (%):")
        months = sorted(results['monthly_performance'].keys())
        monthly_returns = [results['monthly_performance'][month] for month in months]
        min_monthly = min(monthly_returns) if monthly_returns else -5

        for month in months:
            perf = results['monthly_performance'][month]
            print(colorize_output(perf, f"{month}:", 5, min_monthly))
    
    if results['yearly_performance']:
        print("\nYearly Performance (%):")
        years = sorted(results['yearly_performance'].keys())
        for year in years:
            perf = results['yearly_performance'][year]
            print(colorize_output(perf, f"{year}:", 20, -10))
        print(colorize_output(results['annualized_return'], "Annualized Return %:", 25, 10))


# ------------------------------------------------------------------------------
# Optional routines: plotting, buy signal check, and logging summary
# ------------------------------------------------------------------------------

def try_plot_results(cerebro, logger):
    """Attempt to plot the results if the dataset is small enough."""
    try:
        if len(cerebro.datas) <= 10:
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.facecolor'] = '#1e1e1e'
            plt.rcParams['grid.color'] = '#333333'
            
            cerebro.plot(style='candlestick',
                         barup='green',
                         bardown='red',
                         volup='green',
                         voldown='red',
                         grid=True,
                         subplot=True)
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")


def check_buy_signals(logger):
    """Placeholder to check for buy signals (customize as needed)."""
    logger.info("Checking for buy signals... (functionality not implemented)")






def log_summary_results(results, data_feeds, logger):
    """Log a summary of results to the log file."""
    logger.info("=" * 50)
    logger.info("Comprehensive Backtest Results")
    logger.info("=" * 50)
    
    # Basic info
    logger.info(f"Strategy: StockSniper")
    if data_feeds:
        logger.info(f"Period: {len(data_feeds[0][1])} trading days")
    logger.info(f"Stocks analyzed: {len(data_feeds)}")
    logger.info(f"Initial Capital: ${results['initial_value']:.2f}")
    logger.info(f"Final Value: ${results['final_value']:.2f}")
    logger.info(f"Total Return: {results['total_return']:.2f}%")
    logger.info(f"Annualized Return: {results['annualized_return']:.2f}%")
    
    # Risk metrics
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
    logger.info(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    logger.info(f"SQN: {results['sqn_value']:.2f} ({results['sqn_description']})")
    logger.info(f"Gain to Pain: {results['gain_to_pain_ratio']:.2f}")
    logger.info(f"Ulcer Index: {results['ulcer_index']:.2f}")
    logger.info(f"Max Drawdown: {results['max_dd']:.2f}%")
    logger.info(f"Max Drawdown Duration: {results['max_dd_duration']} days")
    
    # Trade statistics
    logger.info(f"Total Trades: {results['total_closed']}")
    logger.info(f"Win Rate: {results['percent_profitable']:.2f}%")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Risk/Reward Ratio: {results['risk_reward_ratio']:.2f}")
    logger.info(f"Average Holding Period: {results['avg_trade_len']:.1f} days")
    logger.info(f"Avg. Winning Trade: ${results['won_avg']:.2f}")
    logger.info(f"Avg. Losing Trade: ${results['lost_avg']:.2f}")
    logger.info(f"Largest Win: ${results['won_max']:.2f}")
    logger.info(f"Largest Loss: ${results['lost_max']:.2f}")
    logger.info(f"Max Consecutive Wins: {results['max_consecutive_wins']}")
    logger.info(f"Max Consecutive Losses: {results['max_consecutive_losses']}")


def create_results_summary(results):
    """Return a summary dictionary of the key backtest results."""
    summary = {
        'total_return': results['total_return'],
        'annualized_return': results['annualized_return'],
        'sharpe_ratio': results['sharpe_ratio'],
        'sortino_ratio': results['sortino_ratio'],
        'calmar_ratio': results['calmar_ratio'],
        'max_drawdown': results['max_dd'],
        'win_rate': results['percent_profitable'],
        'profit_factor': results['profit_factor'],
        'total_trades': results['total_closed'],
        'avg_trade_pnl': results['avg_profit_per_trade'],
        'risk_reward_ratio': results['risk_reward_ratio'],
        'sqn': results['sqn_value']
    }
    return summary



if __name__ == "__main__":
    main()
    run_post_backtest_organization()

