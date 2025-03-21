# STOCK SNIPER: SYSTEM ARCHITECTURE & DATA FLOW

## PIPELINE OVERVIEW

`[0] Dashboard → [0] app → [1] TickerDownloader → [2] BulkPriceDownloader → [3] Indicators → [4] Predictor → [5] NightlyBackTester → [6] DailyBroker`

## COMPONENT BREAKDOWN

0️⃣ Dashboard
**Purpose**: Configure logging
**Execution**: On demand
**Input**: , )
        if not df.empty:
            logging.info(f, )
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            , , )
        return df
    
    except Exception as e:
        logging.error(f, )
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            , )
            return df
        else:
            logging.warning(f, )
        return generate_mock_price_data(symbol, days)




def generate_mock_price_data(symbol, days=100):
    , , )

def create_empty_chart(message):
    , )
        
        if df.empty:
            logging.warning(, , )
        return df
    
    except Exception as e:
        logging.error(f, )
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            , 
**Dependencies**: pandas, numpy, flask, Flask, plotly, flask_socketio, SocketIO, threading, traceback, parquet, the, actual, trade, signals, both, price
**Key Features**: Kill Position, Default Metrics, Create Empty Chart, Load Historical Price Data, Probability Chart

🔄 app
**Purpose**: Configure logging
**Execution**: On demand
**Input**: , )
        if not df.empty:
            logging.info(f, )
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            , , )
        return df
    
    except Exception as e:
        logging.error(f, )
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            , )
            return df
        else:
            logging.warning(f, )
        return generate_mock_price_data(symbol, days)




def generate_mock_price_data(symbol, days=100):
    , , )

def create_empty_chart(message):
    , )
        
        if df.empty:
            logging.warning(, , )
        return df
    
    except Exception as e:
        logging.error(f, )
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            , 
**Dependencies**: pandas, numpy, flask, Flask, plotly, flask_socketio, SocketIO, threading, traceback, parquet, the, actual, trade, signals, both, price
**Key Features**: Kill Position, Default Metrics, Create Empty Chart, Load Historical Price Data, Probability Chart

1️⃣ TickerDownloader
**Purpose**: !/root/root/miniconda4/envs/tf/bin/python
**Execution**: On demand
**Input**: )
    parser.add_argument(, , action=, )
    parser.add_argument(, )
        current_date = datetime.datetime.now().strftime(, , logger=logger):
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount(, )
        try:
            ticker_count = download_and_convert_ticker_cik_file(logger)
            logger.info(f, )
        except Exception as e:
            logger.error(f, )
    else:
        logger.info(, )
    logger.info(
**Output**: )
    args = parser.parse_args()
    return args

def download_and_convert_ticker_cik_file(logger):
    try:
        logger.debug(, )
            
            # Return processed data count for statistics
            return df.shape[0]

    except requests.exceptions.RequestException as e:
        logger.error(f
**Dependencies**: requests, pandas, argparse, HTTPAdapter, urllib3, Retry, Util, get_logger
**Key Features**: Download And Convert Ticker Cik File

2️⃣ BulkPriceDownloader
**Purpose**: 2__BulkPriceDownloader.py - OPTIMIZED VERSION
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
**Execution**: Daily
**Input**: )
PROGRESS_FILE = os.path.join(LOG_DIRECTORY, , )

# OPTIMIZED SETTINGS: Streamlined for performance
# Only download TRADES data (normal OHLCV)
DATA_TYPES = [,  in error_msg.lower():
                await asyncio.sleep(1.0)  # Longer pause for ID conflicts
            else:
                await asyncio.sleep(0.5)  # Brief pause for other errors
                
            # Make sure we, , , )
        return None, fail_reason

def save_to_parquet(df, file_path):
    , , , processed_tickers, )
        logger.error(f, )
        return [], []

def find_latest_ticker_cik_file(directory):
    , )
    
    success_count = len(processed_tickers)
    fail_count = 0
    total_count = len(tickers)
    
    print(f, )
    print(, , 
                unit=, )
    print(, )
    print(, )
    
    # IBKR connection settings
    parser.add_argument(, )
    parser.add_argument(, )
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            print(, )
        print(
**Output**: ⚠️ Too many connection failures. Check your TWS/Gateway settings.
**Dependencies**: argparse, pandas, numpy, asyncio, nest_asyncio, tqdm, traceback, glob, concurrent, ThreadPoolExecutor, random, warnings, collections, deque, functools, lru_cache, Util, get_logger, ib_insync, IB, the, active, a, JSON, existing, IBKR, RF_PREDICTIONS_DIRECTORY, CIK
**Key Features**: Save Progress, Get Existing Tickers, Get Contract, Display Header, Process Ticker

3️⃣ Indicators
**Purpose**: !/root/root/miniconda4/envs/tf/bin/python
**Execution**: On demand
**Input**: : , : 500,
    , ]}, ], f) 
                 for f in os.listdir(CONFIG[, ]) 
                 if f.endswith(
**Output**: : , ])


def process_data_files(run_percent):
    print(f, ], exist_ok=True)
    clear_output_directory(CONFIG[, ])

    file_paths = [os.path.join(CONFIG[
**Dependencies**: pandas, numpy, scipy, linregress, argparse, traceback, pykalman, KalmanFilter, find_peaks, gaussian_kde, argrelextrema, concurrent, ProcessPoolExecutor, numba, njit, entropy, tqdm, watchdog, Observer, FileSystemEventHandler, Util, get_logger, high, the, rolling, Bands
**Key Features**: Calculate Klinger Oscillator, Calculate Genetic Indicators, Dataqualitycheck, Calculate Indicators Numba, Add Kalman And Recurrence Metrics

4️⃣ Predictor
**Purpose**: #predictor script
**Execution**: On demand
**Input**: : , .parquet, )
    
    for file in os.listdir(output_directory):
        if file.endswith(, .parquet, , n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                try:
                    y_pred_proba = clf.predict_proba(X)
                except Exception as e:
                    logging.error(f, input_directory, input_directory
**Output**: : , : , : , : , training_data.parquet, model_output_directory, model_output_directory, model_output_directory, )
    
    # Handle feature importances appropriately for each model type
    try:
        if model_type == , ], index=False)
        logging.info(f, feature_importance_output, )



def main():
    
    
    model_type = config[, data_output_directory, ], model_filename),
            output_directory=config[, ],
            target_column=config[
**Dependencies**: random, pandas, numpy, xgboost, XGBClassifier, sklearn, train_test_split, classification_report, joblib, dump, argparse, precision_recall_curve, tqdm, parallel_backend, contextlib, redirect_stdout, io, Util, get_logger, catboost, CatBoostClassifier, previous, fit, matplotlib, model
**Key Features**: Prepare Training Data, Train Model, Drop String Columns, Predict And Save

5️⃣ NightlyBackTester
**Purpose**: !/usr/bin/env python
**Execution**: Daily
**Input**: UpProbability, )
        traceback.print_exc()

    return None

def parallel_load_data(file_paths, last_trading_date):
    , , 
        ))
    
    return [result for result in results if result is not None]

def read_trading_data():
    , , Correlations.parquet, )

        if , Date, )
        except:
            # Create new DataFrame if file doesn, up_prob, 
**Output**: , )
            return
            
        df = pd.DataFrame(self.trades)
        
        numeric_cols = [
            , )














class StockSniperStrategy(bt.Strategy):
    # Use the parameters from STRATEGY_PARAMS_TUPLE
    params = STRATEGY_PARAMS
    
    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        for d in self.datas:
            self.inds[d][, t meet all criteria., , )
        return
    
    try:
        # Get next trading day if not provided
        if next_trading_day is None:
            current_date = datetime.now().date()
            next_trading_day = get_next_trading_day(current_date)
            logger.info(f, , total_return, annualized_return, final_value, initial_value, sharpe_ratio, sortino_ratio, calmar_ratio, gain_to_pain_ratio, omega_ratio, sqn_value, vwr, enhanced_modified_sqn, )







def print_sqn_quality(results):
    , , max_dd, max_dd_duration, ulcer_index, recovery_factor, common_sense_ratio, risk_of_ruin, daily_volatility, annualized_volatility, var_95, cvar_95, , total_closed, percent_profitable, gross_win_rate, commission_impact_pct, breakeven_threshold_pct, won_avg, lost_avg, avg_win_pct, avg_loss_pct, won_max, lost_max, largest_win_pct, largest_loss_pct, avg_profit_per_trade, profit_factor, net_profit_drawdown_ratio, , avg_trade_len, longest_trade, shortest_trade, max_consecutive_wins, max_consecutive_losses, win_loss_count_ratio, risk_reward_ratio, kelly_percentage, , positive_days_pct, max_pos_streak, max_neg_streak, profit_per_day, {month}:, {year}:, annualized_return
**Dependencies**: argparse, random, pandas, numpy, backtrader, matplotlib, tqdm, pyarrow, multiprocessing, numba, njit, traceback, collections, Counter, pandas_market_calendars, glob, warnings, Util, NYSE, STRATEGY_PARAMS_TUPLE, original, 0, the, file, concurrent, command, STRATEGY_PARAMS, strategy, analyzers, all, each, your, daily
**Key Features**: Get Mean Correlation, Run Strategy Optimization, Add Analyzers, Colorize Output, Sort Buy Candidates

6️⃣ DailyBroker
**Purpose**: !/usr/bin/env python
**Execution**: Daily
**Input**: )
    
    return False



class Colors:
    RESET = , )
            return
            
        # Skip if it
**Dependencies**: random, uuid, threading, traceback, socket, zoneinfo, ZoneInfo, backtrader, backtrader_ib_insync, IBStore, ib_insync, pandas, numpy, exchange_calendars, Util, IB, the, dictionary, current, broker, existing, store, your, positions
**Key Features**: Is Restricted, Create Ib Connection, Close Position, Handle Sell Order, Disconnect Ib Safely

## DATA EVOLUTION
**Raw Tickers** → **OHLCV Data** → **Feature-Rich Indicators** → **ML Predictions** → **Trading Signals** → **Live Positions**

## CRITICAL SYSTEM PARAMETERS

| Parameter | Value | Impact |
|-----------|-------|--------|
| timeout | 10 | |
| cache timeout | 10 | |
| active positions | 0 | |
| timeout | 5.00 | |
| position | 0 | |
| request timeout | 10 | |
| atr threshold multiplier | 2 | |
| volume threshold multiplier | 2 | |
| threshold multiplier | 0.65 | |
| optimal threshold pos | 0.90 | |
| threshold pos | 0.70 | |
| threshold neg | 0.70 | |
| threshold | 0.70 | |
| risk per trade | 1.00 | |
| max position pct | 0.20% | |
| min position pct | 0.05% | |
| risk per share | 0.01 | |
| open positions | 0 | |
| min prob threshold | 0.70 | |
| high threshold | 0.70 | |
| connection timeout | 20.00 | |

## RISK MANAGEMENT APPROACH
1. **Position Sizing**: ATR-based to normalize volatility exposure
2. **Correlation Control**: Maximum allocation per correlation cluster
3. **Rule 201 Monitoring**: Avoid stocks in rapid decline
4. **Active Protection**: Trailing stops with take-profit targets
5. **Timeout Management**: Force-exit positions after timeout period

## SAMPLE DATA FLOW: AAPL
1. Downloaded from SEC list as valid NASDAQ ticker
2. OHLCV data acquired via IBKR for historical prices
3. Technical indicators calculated including ATR, momentum metrics
4. ML model produces upward probability score
5. Backtester validates signal against strategy parameters
6. Position sized according to risk management rules
7. Live trade executed via broker integration

## EXTERNAL DEPENDENCIES
* Interactive Brokers (TWS/Gateway)
* SEC API
* NYSE market calendar
* Pandas, NumPy, Scikit-learn ecosystem
* xgboost library
* catboost library
* sklearn library

## UNIQUE SYSTEM STRENGTHS
* Parallel data acquisition with smart error handling
* Non-lookahead indicator calculation
* ML probability-based signals (not just binary)
* Multi-factor correlation clustering for diversification
