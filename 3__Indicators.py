#!/root/root/miniconda4/envs/tf/bin/python
import os
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
from scipy.stats import linregress
import logging
import argparse
import traceback
from pykalman import KalmanFilter
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, jit
from scipy.stats import entropy
from tqdm import tqdm
from datetime import datetime, timedelta


"""
This script processes financial market data to calculate various technical indicators and saves the processed data Parquet format. It is designed to handle large datasets efficiently by utilizing multiple cores of the processor.

Features:
- Processes data files containing market data like Open, High, Low, Close, Volume, etc.
- Calculates a comprehensive set of technical indicators including moving averages, ATR, VWAP, RSI, and apoxx 50 others.
- Implements anomaly detection by squashing outliers and interpolating missing data.
- Optimized for performance with multithreading and conditional execution based on processor core count.
- Robust error handling and logging to track process efficiency and diagnose issues.
- Flexibility in processing multiple files from a specified directory and saving them in a desired format.

Usage:
- The script reads market data files from a specified input directory, processes each file to compute various indicators, and writes the enhanced data to an output directory.
- The file format is preserved during the process.
- Log information about the process, including processing time and any encountered errors, is saved in a log file.
- It can be customized for different sets of indicators or processing logic based on specific needs.

Example:
- To process all files in the 'Data/PriceData' directory and save the processed files in the 'Data/IndicatorData' directory, simply run the script. The process is automated and requires no additional command-line arguments.

Notes:
- Ensure that the input directory contains valid market data files in supported formats.
- Modify the CONFIG dictionary to specify paths for input/output directories and log file.
- The script utilizes multithreading for faster processing, making it suitable for large datasets.
- Check the log file for detailed information about the processing time and potential errors.
"""


def setup_logging():
    """Set up logging configuration."""
    log_file = CONFIG['log_file']
    log_dir = os.path.dirname(log_file)
    
    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get the root logger
    logger = logging.getLogger()
    if len(logger.handlers) == 0:
        # Set the logging level for the logger
        logger.setLevel(logging.INFO)
        
        # Create a file handler that logs to the specified file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
    
        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
    
        # Add the file handler to the logger
        logger.addHandler(file_handler)
    
    return logger




CONFIG = {
    'input_directory': 'Data/PriceData',
    'output_directory': 'Data/IndicatorData',
    'log_file': os.path.join('Data', 'logging', '3__Indicators.log'),  # Use os.path.join for compatibility
    'log_lines_to_read': 500,
    'core_count_division': True,
}

logger = setup_logging()


##===========================(Indicators)===========================##
##===========================(Indicators)===========================##
##===========================(Indicators)===========================##

def squash_col_outliers(df, col_name=None, num_std_dev=3):
    if col_name:
        columns_to_process = [col_name]
    else:
        columns_to_process = df.select_dtypes(include=['float64']).columns

    # Loop through each column
    for col in columns_to_process:
        if col not in df.columns or df[col].dtype != 'float64':
            continue
        rolled_means = df[col][df[col] != 0].rolling(window=282, min_periods=1).mean()
        rolled_stds = df[col][df[col] != 0].rolling(window=282, min_periods=1).std()
        lower_bounds = rolled_means - num_std_dev * rolled_stds
        upper_bounds = rolled_means + num_std_dev * rolled_stds
        df[col] = df[col].clip(lower=lower_bounds, upper=upper_bounds)
    return df


def interpolate_columns(df, max_gap_fill=50):
    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number):
            continue
        consec_nan_count = df[column].isna().astype(int).groupby(df[column].notna().astype(int).cumsum()).cumsum()
        mask = consec_nan_count <= max_gap_fill
        df.loc[mask, column] = df.loc[mask, column].interpolate(method='linear')
        df[column] = df[column].ffill()
    return df


def find_best_fit_line(x, y):
    try:
        slope, intercept, _, _, _ = linregress(x, y)
        return slope, intercept
    except ValueError:  # Handle any mathematical errors
        return np.nan, np.nan
    

def find_levels(df, window_size):
    def calculate_level_distance(rolling_window):
        # Handle incomplete window
        if len(rolling_window) < window_size:
            return np.nan, np.nan

        # Find top 10 highest and lowest levels
        high_levels = np.argsort(rolling_window)[-10:]
        low_levels = np.argsort(rolling_window)[:10]

        # Calculate lines of best fit
        high_slope, high_intercept = find_best_fit_line(high_levels, rolling_window[high_levels])
        low_slope, low_intercept = find_best_fit_line(low_levels, rolling_window[low_levels])

        # Current position (last index in the rolling window)
        current_position = len(rolling_window) - 1

        # Calculate distance from high and low lines
        high_line_value = high_slope * current_position + high_intercept
        low_line_value = low_slope * current_position + low_intercept

        # Return the distance from high and low lines separately
        distance_from_high = abs(rolling_window[-1] - high_line_value)
        distance_from_low = abs(rolling_window[-1] - low_line_value)

        return distance_from_high, distance_from_low

    distances = df['Close'].rolling(window=window_size).apply(calculate_level_distance, raw=True)
    df['Distance_From_High'] = [dist[0] for dist in distances]
    df['Distance_From_Low'] = [dist[1] for dist in distances]
    return df


def detect_peaks_and_valleys(df, column_name, prominence=1):
    peaks_column = column_name + '_Peaks'
    valleys_column = column_name + '_Valleys'
    
    df[peaks_column] = np.nan
    df[valleys_column] = np.nan

    peaks, properties = find_peaks(df[column_name], prominence=prominence)
    df.loc[peaks, peaks_column] = df[column_name][peaks]

    valleys, properties = find_peaks(-df[column_name], prominence=prominence)
    df.loc[valleys, valleys_column] = df[column_name][valleys]

    return df



def hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0



def rolling_hurst_exponent(series, window_size):
    series_clean = series.dropna()
    def hurst_window(window):
        return hurst_exponent(window)
    return series_clean.rolling(window=window_size).apply(hurst_window, raw=True)




###=======[add_multiple_mean_reversion_z_scores]======###
def add_multiple_mean_reversion_z_scores(data, price_column='Smoothed_Close', windows=[28, 90, 151], std_multipliers=[1, 1, 3]):
    for window, std_multiplier in zip(windows, std_multipliers):
        mean_col = f'Rolling_Mean_{window}'
        std_col = f'Rolling_Std_{window}'
        z_score_col = f'Mean_Reversion_Z_Score_{window}_std_{std_multiplier}'

        rolling_window = data[price_column].rolling(window=window)
        data[mean_col] = rolling_window.mean()
        data[std_col] = rolling_window.std()
        data[z_score_col] = (data[price_column] - data[mean_col]) / (data[std_col] * std_multiplier)

    return data

#==========[calculate_percentage_difference_from_ewma]==========#
@njit
def calculate_ewm(arr, alpha):
    n = len(arr)
    ewm_arr = np.empty(n)
    ewm_arr[0] = arr[0]
    for i in range(1, n):
        ewm_arr[i] = alpha * arr[i] + (1 - alpha) * ewm_arr[i - 1]
    return ewm_arr

@njit
def calculate_percentage_difference_from_ewma_numba(close_prices, periods):
    n = len(close_prices)
    results = np.empty((n, len(periods)))
    
    for j, period in enumerate(periods):
        alpha = 2 / (period + 1)
        ewm_arr = calculate_ewm(close_prices, alpha)
        for i in range(n):
            results[i, j] = (close_prices[i] - ewm_arr[i]) / ewm_arr[i] * 100
    
    return results

def calculate_percentage_difference_from_ewma(data, price_column='Close', periods=[14, 151, 269], adjust=False):
    close_prices = data[price_column].values
    results = calculate_percentage_difference_from_ewma_numba(close_prices, periods)

    for j, period in enumerate(periods):
        ewm_col = f'EWM_{period}'
        pct_diff_col = f'Pct_Diff_EWM_{period}'
        data[ewm_col] = calculate_ewm(close_prices, 2 / (period + 1))
        data[pct_diff_col] = results[:, j]

    return data


def calculate_poc_and_metrics(data, window_size=70):
    @njit(fastmath=True)
    def find_poc(prices, volumes):
        unique_prices = np.unique(prices)
        volume_by_price = np.zeros(len(unique_prices))
        
        for i in range(len(prices)):
            for j in range(len(unique_prices)):
                if prices[i] == unique_prices[j]:
                    volume_by_price[j] += volumes[i]
        
        max_volume_idx = np.argmax(volume_by_price)
        return unique_prices[max_volume_idx]

    @njit(fastmath=True)
    def calculate_poc_rolling(prices, volumes, dates, window_size):
        n = len(prices)
        poc_values = np.empty(n - window_size + 1)
        poc_dates = np.empty(n - window_size + 1, dtype=np.int64)
        
        for i in range(n - window_size + 1):
            window_prices = prices[i:i + window_size]
            window_volumes = volumes[i:i + window_size]
            poc = find_poc(window_prices, window_volumes)
            poc_values[i] = poc
            poc_dates[i] = dates[i + window_size - 1]
        
        return poc_values, poc_dates

    close_prices = data['Close'].values
    volumes = data['Volume'].values
    dates = data['Date'].astype(np.int64).values

    poc_values, result_dates = calculate_poc_rolling(close_prices, volumes, dates, window_size)

    poc_df = pd.DataFrame({'Date': pd.to_datetime(result_dates), 'PoC': poc_values})
    data = pd.merge(data, poc_df, on='Date', how='left')

    if 'PoC' in data.columns:
        data['Pct_Diff_PoC'] = (data['Close'] - data['PoC']) / data['PoC'] * 100
        data['PoC_Mean'] = data['PoC'].rolling(window=window_size).mean()
        data['PoC_SD'] = data['PoC'].rolling(window=window_size).std()
        data['Pct_Diff_PoC_Mean'] = data['Pct_Diff_PoC'].rolling(window=window_size).mean()
        data['Pct_Diff_PoC_SD'] = data['Pct_Diff_PoC'].rolling(window=window_size).std()
    else:
        logging.error("PoC column not found after calculation.")

    return data



def add_complexity_metrics(df, window_size=90):
    def calculate_complexity_invariant_distance(local_data):
        rolling_variance = local_data.rolling(window=window_size).var()
        complexity_distance = rolling_variance.diff().abs()
        return complexity_distance

    df['Complexity_Invariant_Distance'] = calculate_complexity_invariant_distance(df['Close'])
    df['CID_Mean'] = df['Complexity_Invariant_Distance'].rolling(window=window_size).mean()
    df['CID_SD'] = df['Complexity_Invariant_Distance'].rolling(window=window_size).std()
    df['CID_Diff_From_Mean'] = df['Complexity_Invariant_Distance'] - df['CID_Mean']

    return df



def add_kalman_and_recurrence_metrics(df, epsilon_multiplier=0.01, window_size=70):
    def apply_kalman_filter(close_prices):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=close_prices.iloc[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(close_prices)
        return state_means.flatten()

    @jit(nopython=True, fastmath=True)
    def calculate_recurrence_rate(data, epsilon):
        n = len(data)
        recurrences = np.zeros(n - window_size + 1)
        for i in range(n - window_size + 1):
            window = data[i:i + window_size]
            count = 0
            for j in range(window_size):
                for k in range(j + 1, window_size):
                    if np.abs(window[j] - window[k]) < epsilon:
                        count += 1
            recurrences[i] = count / (window_size * (window_size - 1) / 2)
        return recurrences

    df['Smoothed_Close'] = apply_kalman_filter(df['Close'])
    epsilon = epsilon_multiplier * np.std(df['Smoothed_Close'])
    df['Recurrence_Rate'] = np.nan
    recurrences = calculate_recurrence_rate(df['Smoothed_Close'].values, epsilon)
    df.loc[df.index[window_size-1:window_size+len(recurrences)-1], 'Recurrence_Rate'] = recurrences

    return df



def add_kalman_and_entropy_metrics(df, window_size=70, bins=30):
    def apply_kalman_filter(close_prices):
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=close_prices.iloc[0],
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=0.01)
        state_means, _ = kf.filter(close_prices)
        return state_means.flatten()

    def calculate_differences_entropy(smoothed_prices):
        log_returns = np.log(smoothed_prices / np.roll(smoothed_prices, 1))[1:]
        entropies = []
        for i in range(len(log_returns) - window_size + 1):
            window = log_returns[i:i + window_size]
            hist, _ = np.histogram(window, bins=bins, density=True)
            entropies.append(entropy(hist, base=2))
        return entropies

    df['Smoothed_Close'] = apply_kalman_filter(df['Close'])
    df['Entropy of Differences'] = np.nan
    entropy_values = calculate_differences_entropy(df['Smoothed_Close'])
    df.loc[df.index[window_size-1:window_size+len(entropy_values)-1], 'Entropy of Differences'] = entropy_values

    return df




def calculate_ema_volume_change(df, window=90, ema_span=20):

    # Calculate rolling median and IQR
    rolling_median = df['Volume'].rolling(window=window).median()
    rolling_iqr = df['Volume'].rolling(window=window).apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    
    # Scale the volume
    df['Volume_Scaled'] = (df['Volume'] - rolling_median) / rolling_iqr
    df['Volume_Scaled'] = df['Volume_Scaled'].fillna(0)  # Fill NaNs that may arise from rolling window
    
    # Calculate EMA of the scaled volume
    df['Volume_EMA'] = df['Volume_Scaled'].ewm(span=ema_span, adjust=False).mean()
    
    # Calculate the percentage change of the EMA of the scaled volume
    df['Volume_EMA_Change'] = df['Volume_EMA'].pct_change()
    
    return df['Volume_EMA_Change']



def calculate_parabolic_SAR(df):
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Initialize the SAR with the first row's low value
    sar = low[0]
    # Initial values for High and Low points
    ep = high[0]
    af = 0.02
    sar_values = [sar]

    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if close[i] > close[i - 1]:
            af = min(af + 0.02, 0.2)
        else:
            af = 0.02

        if close[i] > close[i - 1]:
            ep = max(high[i], ep)
        else:
            ep = min(low[i], ep)

        sar = min(sar, low[i], low[i - 1]) if close[i] > close[i - 1] else max(sar, high[i], high[i - 1])
        sar_values.append(sar)

    df['Parabolic_SAR'] = sar_values
    return df



def calculate_apz(data, ema_period=20, atr_period=14, atr_multiplier=None, volatility_factor=0.1):
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    close = data['Close']
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=atr_period).mean()

    if atr_multiplier is None:
        atr_multiplier = volatility_factor * data['Close'].rolling(window=atr_period).std()

    data['APZ_Upper'] = data['EMA'] + (data['ATR'] * atr_multiplier)
    data['APZ_Lower'] = data['EMA'] - (data['ATR'] * atr_multiplier)

    data['APZ_Upper%'] = ((data['APZ_Upper'] - close) / close) * 100
    data['APZ_Lower%'] = ((close - data['APZ_Lower']) / close) * 100
    ##drop the upper and lower apz columns
    data = data.drop(columns=['APZ_Upper', 'APZ_Lower'])
    return data


def calculate_vama(df, price_col='Close', min_period=10, max_period=30):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    volatility = df[price_col].pct_change().rolling(window=21).std()
    normalized_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min())
    window_size = (normalized_vol * (max_period - min_period) + min_period).clip(min_period, max_period).round().astype(int)
    try:
        vama = df[price_col].rolling(window=window_size, min_periods=min_period).mean()
    except Exception as e:
        logging.error(f"Error in VAMA calculation: {e}")
        vama = pd.Series(index=df.index)

    return vama

def add_vama_changes(df, vama, periods):
    for period in periods:
        label = f'VAMA_pct_change_{period}_days'
        df[label] = vama.pct_change(periods=period)
    return df


def compute_VPT(df):
    close_shift_1 = df['Close'].shift(1)
    low_shift_1 = df['Low'].shift(1)
    high_shift_1 = df['High'].shift(1)

    # Calculate conditions
    is_gap_up = (df['Low'] - high_shift_1) / high_shift_1 * 100 > 0.5  # Assuming gap_threshold_percent is 0.5
    is_gap_down = (df['High'] - low_shift_1) / close_shift_1 * 100 < -0.5

    # Calculate new columns
    move_from_gap_percent = np.where(is_gap_up, (df['Close'] - low_shift_1) / low_shift_1 * 100, 0)
    VPT = df['Volume'] * ((df['Close'] - close_shift_1) / close_shift_1)

    # Store calculations in a temporary DataFrame
    temp_df = pd.DataFrame({
        'is_gap_up': is_gap_up,
        'is_gap_down': is_gap_down,
        'move_from_gap%': move_from_gap_percent,
        'VPT': VPT
    }, index=df.index)

    # Concatenate this DataFrame with the original one
    df = pd.concat([df, temp_df], axis=1)
    return df


def AtrVolume(df):
    df['ATR_std'] = df['ATR'].rolling(window=60).std()
    df['Volume_std'] = df['Volume'].rolling(window=60).std()
    ATR_threshold_multiplier = 2  # Example value, can be adjusted
    Volume_threshold_multiplier = 2  # Example value, can be adjusted
    df['ATR_trigger'] = df['ATR'] > (df['ATR_std'] * ATR_threshold_multiplier)
    df['Volume_trigger'] = df['Volume'] > (df['Volume_std'] * Volume_threshold_multiplier)
    df['Oscillator'] = (df['ATR_trigger'] & df['Volume_trigger']).astype(int)
    reset_points = df['Oscillator'].diff().eq(-1).cumsum()
    # Calculate the trigger counter
    df['Trigger_Counter'] = df['Oscillator'].groupby(reset_points).cumsum()
    return df


def ATR_Based_Adaptive_Trend_Channels(df):
    # 200-day Moving Average
    df['200MA'] = df['Close'].rolling(window=200).mean()
    df['14Day_Avg_ATR'] = df['ATR'].rolling(window=14).mean()
    df['Upper_Band'] = df['200MA'] + df['14Day_Avg_ATR']
    df['Lower_Band'] = df['200MA'] - df['14Day_Avg_ATR']
    # Percentage Deviation from Bands
    df['Pct_Deviation_Upper'] = np.where(df['Close'] > df['Upper_Band'],
                                         (df['Close'] - df['Upper_Band']) / df['Upper_Band'] * 100,
                                         -((df['Upper_Band'] - df['Close']) / df['Close']) * 100)

    df['Pct_Deviation_Lower'] = np.where(df['Close'] < df['Lower_Band'],
                                         -((df['Close'] - df['Lower_Band']) / df['Lower_Band']) * 100,
                                         ((df['Lower_Band'] - df['Close']) / df['Close']) * 100)
    ##drop the intermediate columns used to calculate the bands
    df = df.drop(columns=['200MA', '14Day_Avg_ATR', 'Upper_Band', 'Lower_Band'])
    return df



def calculate_klinger_oscillator(df, short_period=34, long_period=55, signal_period=13):
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    typical_price = (high + low + close) / 3
    dm = typical_price.diff()
    dm[dm < 0] = 0

    cmf = volume * dm
    cmf_short = cmf.rolling(window=short_period).sum()
    cmf_long = cmf.rolling(window=long_period).sum()

    kvo = cmf_short - cmf_long
    kvo_signal = kvo.rolling(window=signal_period).mean()

    df['KVO'] = kvo
    df['KVO_Signal'] = kvo_signal

    return df


##======[add_rolling_lzc]======##
@njit
def calculate_percentage_moves(close_prices):
    return (close_prices[1:] - close_prices[:-1]) / close_prices[:-1] * 100

@njit
def bin_percentage_changes(percentage_changes, bin_size=1):
    return np.floor(percentage_changes / bin_size).astype(int)

@njit
def lempel_ziv_complexity(sequence):
    n = len(sequence)
    sub_strings = []
    i, k, l = 0, 1, 1
    while k + l <= n:
        if tuple(sequence[i:i + l]) == tuple(sequence[k:k + l]):
            l += 1
            if k + l > n:
                sub_strings.append(tuple(sequence[i:k + l - 1]))
        else:
            sub_strings.append(tuple(sequence[i:k + l - 1]))
            i = k
            k += 1
            l = 1
    sub_strings.append(tuple(sequence[i:k + l - 1]))
    return len(set(sub_strings))

@njit
def calculate_lzc(binned_changes, window_size):
    n = len(binned_changes)
    lzc_values = np.full(n, np.nan)
    for i in range(n - window_size + 1):
        window = binned_changes[i:i + window_size]
        lzc_values[i + window_size - 1] = lempel_ziv_complexity(window)
    return lzc_values

def add_rolling_lzc(df, window_size=50, bin_size=1):
    close_prices = df['Close'].to_numpy()
    percentage_moves = calculate_percentage_moves(close_prices)
    binned_changes = bin_percentage_changes(percentage_moves, bin_size)

    # Calculate Lempel-Ziv Complexity on the binned sequence for each rolling window
    lzc_values = calculate_lzc(binned_changes, window_size)

    # Add the LZC values to the DataFrame
    df['Lempel_Ziv_Complexity'] = np.concatenate([[np.nan] * (window_size - 1), lzc_values[window_size - 1:]])  # Ensure same length as original data

    return df



def calculate_poc_and_metrics(data, window_size=70):
    @njit
    def find_poc(prices, volumes):
        volume_by_price = {}
        for price, volume in zip(prices, volumes):
            if price in volume_by_price:
                volume_by_price[price] += volume
            else:
                volume_by_price[price] = volume
        max_volume = -1
        poc = -1
        for price, volume in volume_by_price.items():
            if volume > max_volume:
                max_volume = volume
                poc = price
        return poc

    def calculate_poc_rolling(local_data):
        poc_values = []
        dates = []
        for i in range(len(local_data) - window_size + 1):
            window = local_data.iloc[i:i + window_size]
            prices = window['Close'].values
            volumes = window['Volume'].values
            poc = find_poc(prices, volumes)
            poc_values.append(poc)
            dates.append(window['Date'].iloc[-1])
        return pd.DataFrame({'Date': dates, 'PoC': poc_values})

    poc_df = calculate_poc_rolling(data)
    data = pd.merge(data, poc_df, on='Date', how='left')

    if 'PoC' in data.columns:
        data['Pct_Diff_PoC'] = (data['Close'] - data['PoC']) / data['PoC'] * 100
        data['PoC_Mean'] = data['PoC'].rolling(window=window_size).mean()
        data['PoC_SD'] = data['PoC'].rolling(window=window_size).std()
        data['Pct_Diff_PoC_Mean'] = data['Pct_Diff_PoC'].rolling(window=window_size).mean()
        data['Pct_Diff_PoC_SD'] = data['Pct_Diff_PoC'].rolling(window=window_size).std()
    else:
        logging.error("PoC column not found after calculation.")

    return data



@njit
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_cov = np.sum((x - x_mean) * (y - y_mean))
    xx_cov = np.sum((x - x_mean) ** 2)
    slope = xy_cov / xx_cov
    intercept = y_mean - slope * x_mean
    return slope, intercept

@njit
def calculate_indicators_numba(volume, close, high, low):
    n = len(volume)
    volume_lag_1 = np.zeros(n)
    volume_lag_2 = np.zeros(n)
    volume_lag_3 = np.zeros(n)
    volume_rolling_28 = np.zeros(n)
    volume_percent = np.zeros(n)
    volume_std = np.zeros(n)
    volume_slope = np.zeros(n)
    volume_rolling_90 = np.zeros(n)
    volume_percent_rolling_90 = np.zeros(n)
    ado = np.zeros(n)
    ado_close_cor = np.zeros(n)

    # Calculate volume lag
    for i in range(1, n):
        volume_lag_1[i] = volume[i - 1]
        if i > 1:
            volume_lag_2[i] = volume[i - 2]
        if i > 2:
            volume_lag_3[i] = volume[i - 3]

    # Calculate rolling mean, percentage change, and std
    for i in range(28, n):
        volume_rolling_28[i] = np.mean(volume[i-28:i])
        if volume_rolling_28[i] != 0:
            volume_percent[i] = ((volume[i] - volume_rolling_28[i]) / volume_rolling_28[i]) * 100
        volume_std[i] = np.std(volume[i-28:i])

    for i in range(90, n):
        volume_rolling_90[i] = np.mean(volume[i-90:i])
        if volume_rolling_90[i] != 0:
            volume_percent_rolling_90[i] = ((volume[i] - volume_rolling_90[i]) / volume_rolling_90[i]) * 100

    # Calculate volume slope
    for i in range(5, n):
        x = np.arange(5)
        y = volume[i-5:i]
        slope, _ = linear_regression(x, y)
        volume_slope[i] = slope

    # Calculate ADO
    for i in range(n):
        if high[i] != low[i]:  # Avoid division by zero
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            ado[i] = clv * volume[i]

    ado_cumsum = np.cumsum(ado)

    # Calculate rolling correlation
    for i in range(28, n):
        if np.std(close[i-28:i]) > 0 and np.std(ado_cumsum[i-28:i]) > 0:  # Ensure valid standard deviations
            ado_close_cor[i] = np.corrcoef(close[i-28:i], ado_cumsum[i-28:i])[0, 1]

    return (volume_lag_1, volume_lag_2, volume_lag_3, volume_rolling_28, volume_percent,
            volume_std, volume_slope, volume_rolling_90, volume_percent_rolling_90,
            ado_cumsum, ado_close_cor)


def VolumeADO(df):
    # Extract relevant columns as numpy arrays
    volume = df['Volume'].to_numpy()
    close = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()

    # Call the numba optimized function
    (volume_lag_1, volume_lag_2, volume_lag_3, volume_rolling_28, volume_percent,
     volume_std, volume_slope, volume_rolling_90, volume_percent_rolling_90,
     ado_cumsum, ado_close_cor) = calculate_indicators_numba(volume, close, high, low)

    # Add results back to the DataFrame
    df['Volume_lag_1'] = volume_lag_1
    df['Volume_lag_2'] = volume_lag_2
    df['Volume_lag_3'] = volume_lag_3
    df['Volume_rolling_28'] = volume_rolling_28
    df['Volume%'] = volume_percent
    df['Volume_std'] = volume_std
    df['Volume_slope'] = volume_slope
    df['Volume_rolling_90'] = volume_rolling_90
    df['Volume%_rolling_90'] = volume_percent_rolling_90
    df['ADO'] = ado_cumsum
    df['ADOCloseCor'] = ado_close_cor

    return df


def calculate_cmf(df, period=20):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return df


def imminent_channel_breakout(df, ma_period=200, atr_period=14):
    df['200MA'] = df['Close'].rolling(window=ma_period).mean()
    df['14DayATR'] = df['ATR'].rolling(window=atr_period).mean()
    df['Upper_Band'] = df['200MA'] + df['14DayATR']
    df['Lower_Band'] = df['200MA'] - df['14DayATR']
    df['Upper_Band_Slope'] = df['Upper_Band'].diff() / atr_period
    df['Lower_Band_Slope'] = df['Lower_Band'].diff() / atr_period
    df['Close_Momentum'] = df['Close'].diff()
    def calculate_breakout_score(row):
        upper_proximity = 100 * (1 - min(row['Close'] / row['Upper_Band'], 1))
        lower_proximity = 100 * (1 - min(row['Lower_Band'] / row['Close'], 1))
        slope_strength = 50 * (abs(row['Upper_Band_Slope']) + abs(row['Lower_Band_Slope']))
        if row['14DayATR'] == 0:
            momentum_strength = 0
        else:
            momentum_strength = 50 * abs(row['Close_Momentum']) / row['14DayATR']
        return min(100, upper_proximity * 0.4 + lower_proximity * 0.4 + slope_strength * 0.1 + momentum_strength * 0.1)
    df['Breakout_Score'] = df.apply(calculate_breakout_score, axis=1)
    df.drop(['Upper_Band_Slope', 'Lower_Band_Slope', 'Close_Momentum', '14DayATR'], axis=1, inplace=True)
    return df


def calculate_ad_line(df):
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    ad = clv * df['Volume']
    df['AD_Line'] = ad.cumsum()
    return df

def calculate_nvi(df):
    nvi = np.where(df['Volume'] < df['Volume'].shift(1), df['Close'].pct_change(), 0)
    df['NVI'] = (1 + nvi).cumprod()
    return df

def calculate_emv(df, period=14):
    distance_moved = ((df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2)
    box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
    emv = distance_moved / box_ratio
    emv_ma = emv.rolling(window=period).mean()
    df['EMV'] = emv_ma
    return df




def safe_divide(a, b, fill_value=0):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        # Ensure a and b have the same index
        a, b = a.align(b, fill_value=fill_value)
    elif isinstance(a, pd.Series):
        b = pd.Series(b, index=a.index)
    elif isinstance(b, pd.Series):
        a = pd.Series(a, index=b.index)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, pd.Series):
            result = result.where((b != 0) & (b.notna()), fill_value)
        else:
            result = np.where((b != 0) & (~np.isnan(b)), result, fill_value)
    return result


def safe_log(x, epsilon=1e-14):
    return np.log(np.maximum(x, epsilon))



##====================================[GENETIC INDICATORS]====================================##
##====================================[GENETIC INDICATORS]====================================##
##====================================[GENETIC INDICATORS]====================================##
##====================================[GENETIC INDICATORS]====================================##
##====================================[GENETIC INDICATORS]====================================##
##====================================[GENETIC INDICATORS]====================================##

def calculate_genetic_indicators(df):
    epsilon = 1e-10
    
    for i in range(1, 8):
        df[f'High_Lag{i}'] = df['High'].shift(i) + epsilon
        df[f'Low_Lag{i}'] = df['Low'].shift(i) + epsilon
        df[f'Volume_Lag{i}'] = df['Volume'].shift(i) + epsilon
        df[f'Open_Lag{i}'] = df['Open'].shift(i) + epsilon

    df['G_Momentum_Confluence_Indicator'] = safe_divide(df['High_Lag2'], df['High_Lag2'] * df['Open'])
    
    df['G_Price_Gap_Analyzer'] = safe_divide(safe_log(df['Open_Lag2']), df['High_Lag1'])
    
    df['G_Triple_High_Trend_Indicator'] = safe_divide(df['High_Lag2'], df['High_Lag1'] * df['High'])
    
    df['G_Cyclical_Price_Oscillator'] = safe_divide(
        safe_divide(np.cos(safe_divide(df['High_Lag2'], df['High'])), df['High_Lag1']),
        np.sqrt(df['Open_Lag1'] + epsilon)
    )
    
    df['G_Volume_Adjusted_Price_Indicator'] = safe_divide(df['High_Lag2'], safe_divide(df['Volume'], df['Low_Lag1']))

    df['G_Adjusted_Close_Tracker'] = df['Adj Close']
    
    df['G_Volume_Weighted_High_Ratio'] = safe_divide(safe_divide(df['High'], df['High_Lag1']), safe_log(df['Volume'] + 1))
    
    df['G_High_Price_Momentum_Indicator'] = safe_divide(df['High'], (df['High_Lag1'] + df['High_Lag2']) / 2)
    
    df['G_Advanced_Trend_Synthesizer'] = (
        safe_log(safe_divide(df['High_Lag1'] + df['High_Lag5'], df['High_Lag1'])) *
        np.abs(safe_log(safe_divide(df['High_Lag2'], df['High_Lag2'])) - df['High_Lag7']) *
        safe_divide(df['Open_Lag2'], df['Close'])
    )
    
    df['G_Price_Volatility_Gauge'] = safe_divide(np.abs(df['High_Lag2'] - df['High']), df['Open'])
    
    df['G_Multi_Point_Price_Analyzer'] = np.abs(
        safe_divide(
            safe_divide(safe_log(safe_divide(df['High'], df['High_Lag2'])), safe_divide(df['Close'], df['High_Lag2'])),
            safe_divide(df['Close'], df['High_Lag2']) * safe_divide(df['Close'], df['High_Lag1'])
        )
    )
    
    df['G_Logarithmic_Trend_Detector'] = -safe_log(safe_divide(df['High_Lag2'], df['High_Lag4']))
    
    df['G_Complex_Price_Pattern_Indicator'] = safe_log(
        safe_divide(
            np.sqrt(np.sqrt(np.sqrt(np.sqrt(df['High_Lag7'] * df['High_Lag4'] + epsilon)))),
            df['Close']
        )
    )
    
    df['G_Log_Scaled_Price_Ratio'] = safe_log(safe_divide(df['High'], (df['High_Lag1'] + df['High_Lag2']) / 2))
    
    df['G_Volume_Price_Impact_Indicator'] = safe_divide(
        -df['High_Lag1'] + safe_divide(df['High_Lag3'], df['Close']),
        df['Volume']
    )
    
    df['G_Volume_Trend_Analyzer'] = safe_log(safe_divide(df['Volume'], df['Volume_Lag1']))
    
    df['G_Price_Open_Ratio_Indicator'] = safe_divide(
        safe_divide(safe_log(safe_divide(df['High_Lag2'], df['High'])), safe_divide(df['High'], df['Open_Lag2'])),
        safe_divide(df['High'], df['Open_Lag2'])
    )
    
    df['G_Price_Differential_Analyzer'] = (0.1673 / (df['High'] + epsilon) - df['Low']) / (df['High'] + epsilon)
    
    df['G_Lagged_Price_Volume_Convergence'] = safe_divide(df['Low_Lag2'], (df['High_Lag2'] + safe_divide(0.791, df['Low'] * df['High'])))
    
    df['G_Price_Volume_Disparity_Index'] = np.abs(safe_divide(df['Low_Lag2'], df['High_Lag2'])) / (safe_divide(df['High_Lag5'], df['Low_Lag5']) / -0.2831)
    
    df['G_Price_Volatility_Trend_Measure'] = 0.278 - np.abs(safe_divide(df['Low'], df['High']) / safe_divide(df['High_Lag5'], df['Low_Lag5']))

    for i in range(1, 8):
        df = df.drop(columns=[f'High_Lag{i}', f'Low_Lag{i}', f'Volume_Lag{i}', f'Open_Lag{i}'])

    return df





def calculate_kalman_support_resistance(df, window=140):
    """Calculate Kalman filter and support/resistance levels in a non-fragmented way"""
    try:
        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=df['Close'].values[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=.01
        )
        
        # Calculate all values first
        kalman_values = kf.filter(df['Close'].values)[0].flatten()  # Flatten the output array
        
        # Find extrema indices
        minima_idx = argrelextrema(kalman_values, np.less_equal, order=window)[0]
        maxima_idx = argrelextrema(kalman_values, np.greater_equal, order=window)[0]
        
        # Create arrays of NaN with same length as df
        minima_values = np.full(len(df), np.nan)
        maxima_values = np.full(len(df), np.nan)
        
        # Fill in the extrema values safely
        if len(minima_idx) > 0:
            minima_values[minima_idx] = kalman_values[minima_idx]
        if len(maxima_idx) > 0:
            maxima_values[maxima_idx] = kalman_values[maxima_idx]
        
        # Forward fill the values
        minima_values = pd.Series(minima_values).ffill().values
        maxima_values = pd.Series(maxima_values).ffill().values
        
        # Calculate support and resistance percentages safely
        with np.errstate(divide='ignore', invalid='ignore'):
            support_pct = np.where(
                minima_values > 0,
                (df['Close'].values - minima_values) / minima_values * 100,
                np.nan
            )
            resistance_pct = np.where(
                df['Close'].values > 0,
                (maxima_values - df['Close'].values) / df['Close'].values * 100,
                np.nan
            )
        
        # Handle conversion window
        conversion_window = 5
        for i in range(conversion_window, len(df)):
            if all(kalman_values[i-conversion_window:i+1] > maxima_values[i]):
                minima_values[i] = maxima_values[i]
                maxima_values[i] = np.nan
            if all(kalman_values[i-conversion_window:i+1] < minima_values[i]):
                maxima_values[i] = minima_values[i]
                minima_values[i] = np.nan
        
        # Create new DataFrame with the calculated values
        new_columns = {
            'Kalman': kalman_values,
            'minima': minima_values,
            'maxima': maxima_values,
            'Distance to Support (%)': support_pct,
            'Distance to Resistance (%)': resistance_pct
        }
        
        return pd.DataFrame(new_columns, index=df.index)
    
    except Exception as e:
        logging.error(f"Error in calculate_kalman_support_resistance: {str(e)}")
        # Return empty DataFrame with expected columns if calculation fails
        return pd.DataFrame({
            'Kalman': np.full(len(df), np.nan),
            'minima': np.full(len(df), np.nan),
            'maxima': np.full(len(df), np.nan),
            'Distance to Support (%)': np.full(len(df), np.nan),
            'Distance to Resistance (%)': np.full(len(df), np.nan)
        }, index=df.index)








def calculate_rolling_indicators(df, pct_change_close, rolling_14, rolling_20):
    """Calculate all rolling and percentage-based indicators at once"""
    new_columns = {
        'percent_change_Close': pct_change_close,
        'pct_change_std': rolling_20.std(),
        'percent_change_Close_lag_1': pct_change_close.shift(1),
        'percent_change_Close_lag_3': pct_change_close.shift(3),
        'percent_change_Close_lag_5': pct_change_close.shift(5),
        'percent_change_Close_lag_10': pct_change_close.shift(10),
        'pct_change_std_rolling': rolling_20.mean(),
    }

    # Abnormal percentage change
    threshold_multiplier = 0.65
    abnormal_pct_change_threshold = rolling_20.mean() + threshold_multiplier * new_columns['pct_change_std']
    new_columns['days_since_abnormal_pct_change'] = (pct_change_close > abnormal_pct_change_threshold).cumsum()

    # Direction flipper calculations
    direction_flipper = (pct_change_close > 0).astype(int)
    new_columns['direction_flipper_count5'] = direction_flipper.rolling(window=5).sum()
    new_columns['direction_flipper_count_10'] = direction_flipper.rolling(window=10).sum()
    new_columns['direction_flipper_count_14'] = direction_flipper.rolling(window=14).sum()

    # Keltner Channel calculations
    keltner_central = df['Close'].ewm(span=20).mean()
    keltner_range = df['ATR'] * 1.5
    new_columns['KC_UPPER%'] = ((keltner_central + keltner_range) - df['Close']) / df['Close'] * 100
    new_columns['KC_LOWER%'] = (df['Close'] - (keltner_central - keltner_range)) / df['Close'] * 100

    # VWAP divergence
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    new_columns['VWAP_Divergence'] = df['Close'] - vwap
    new_columns['EFI'] = df['Close'].diff() * df['Volume']

    return pd.DataFrame(new_columns, index=df.index)













def indicators(df):
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Volume'] = df['Volume'].ffill()

    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    # Calculate RSI first to ensure it exists for divergence calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14, min_periods=1).mean() 
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate all other indicators
    df = calculate_genetic_indicators(df)
    calculate_apz(df)
    df = calculate_parabolic_SAR(df)
    df = ATR_Based_Adaptive_Trend_Channels(df)
    df = imminent_channel_breakout(df)
    df = calculate_klinger_oscillator(df)
    df = calculate_cmf(df)
    df = calculate_ad_line(df)
    df = calculate_nvi(df)
    df = calculate_emv(df)
    df = VolumeADO(df)

    close_shift_1 = close.shift(1)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - close_shift_1), np.abs(close_shift_1 - low)))
    window = 14
    rolled = np.lib.stride_tricks.sliding_window_view(true_range, window)
    mean_rolled = np.mean(rolled, axis=1)
    df['ATR'] = np.pad(mean_rolled, (window-1, 0), 'constant', constant_values=(np.nan,))
    df['ATR%'] = (df['ATR'] / close) * 100
    df['ATR%_change'] = df['ATR%'].pct_change()
    df = AtrVolume(df)

    # Moving averages
    df['200ma'] = close.rolling(window=200).mean()
    df['14ma'] = close.rolling(window=14).mean()
    df['14ma%'] = ((close - df['14ma']) / df['14ma']) * 100
    df['200ma%'] = ((close - df['200ma']) / df['200ma']) * 100
    df['SMA_200'] = close.rolling(window=200).mean()
    df['SMA_14'] = close.rolling(window=14).mean()
    df['std_14'] = close.rolling(window=14).std()
    df['Std_Devs_from_SMA'] = (df['SMA_200'] - df['SMA_14']) / df['std_14']
    df['14ma-200ma'] = df['14ma'] - df['200ma']
    df['14ma%_change'] = df['14ma%'].pct_change()
    df['14ma%_count'] = df['14ma%'].gt(0).rolling(window=14).sum()
    df['200ma%_count'] = df['200ma%'].gt(0).rolling(window=200).sum()
    df['14ma_crossover'] = (close > df['14ma'])
    df['200ma_crossover'] = (close > df['200ma'])
    df['200DAY_ATR'] = df['200ma'] + df['ATR']
    df['200DAY_ATR-'] = df['200ma'] - df['ATR']
    df['200DAY_ATR%'] = df['200DAY_ATR'] / close
    df['200DAY_ATR-%'] = df['200DAY_ATR-'] / close
    df['percent_from_high'] = ((close - close.cummax()) / close.cummax()) * 100
    df['new_high'] = (close == close.cummax())
    df['days_since_high'] = (~df['new_high']).cumsum() - (~df['new_high']).cumsum().where(df['new_high']).ffill().fillna(0)
    df['percent_range'] = (high - low) / close * 100

    # VWAP calculations
    typical_price = (high + low + close) / 3
    df['VWAP'] = (typical_price * volume).rolling(window=14).sum() / volume.rolling(window=14).sum()
    df['VWAP_std14'] = df['VWAP'].rolling(window=14).std()
    df['VWAP_std200'] = df['VWAP'].rolling(window=20).std()
    df['VWAP%'] = ((close - df['VWAP']) / df['VWAP']) * 100
    df['VWAP%_from_high'] = ((df['VWAP'] - close.cummax()) / close.cummax()) * 100
    
    # OBV
    obv_condition = df['Close'] > close_shift_1
    df['OBV'] = np.where(obv_condition, volume, -volume).cumsum()

    # Weighted Close Price Change Velocity
    window = 10
    price_change = close.diff().fillna(0)
    weights = np.linspace(1, 0, window)
    weights /= np.sum(weights)
    weighted_velocity = price_change.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    df['Weighted_Close_Change_Velocity'] = weighted_velocity

    pct_change_close = close.pct_change()
    rolling_20 = df['Close'].rolling(window=20)
    rolling_14 = df['Close'].rolling(window=14)

    # Calculate all rolling indicators at once
    rolling_indicators = calculate_rolling_indicators(df, pct_change_close, rolling_14, rolling_20)
    df = pd.concat([df, rolling_indicators], axis=1)

    # Abnormal percentage change calculation
    threshold_multiplier = 0.65
    abnormal_pct_change_threshold = rolling_20.mean() + threshold_multiplier * df['pct_change_std']
    df['days_since_abnormal_pct_change'] = (pct_change_close > abnormal_pct_change_threshold).cumsum()

    # VWAP Divergence
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Divergence'] = df['Close'] - vwap
    df['EFI'] = df['Close'].diff() * df['Volume']

    # Direction flipper calculations
    df['direction_flipper'] = (pct_change_close > 0).astype(int)
    df['direction_flipper_count5'] = df['direction_flipper'].rolling(window=5).sum()
    df['direction_flipper_count_10'] = df['direction_flipper'].rolling(window=10).sum()
    df['direction_flipper_count_14'] = df['direction_flipper'].rolling(window=14).sum()
    df = df.drop(columns=['direction_flipper'])

    # ATR and Keltner Channels
    df['ATR'] = rolling_14.apply(lambda x: np.mean(np.abs(np.diff(x))))
    keltner_central = df['Close'].ewm(span=20).mean()
    keltner_range = df['ATR'] * 1.5
    df['KC_UPPER%'] = ((keltner_central + keltner_range) - df['Close']) / df['Close'] * 100
    df['KC_LOWER%'] = (df['Close'] - (keltner_central - keltner_range)) / df['Close'] * 100

    # Calculate Kalman filter and support/resistance using the new function
    kalman_df = calculate_kalman_support_resistance(df)
    df = pd.concat([df, kalman_df], axis=1)

    # Additional Kalman calculations
    epsilon = 0.001
    df['Perturbed_Kalman'] = df['Kalman'] * (1 + epsilon)
    df['Divergence'] = np.abs(df['Perturbed_Kalman'] - df['Kalman'])
    df['Log_Divergence'] = np.log(df['Divergence'] + np.finfo(float).eps)
    df['Lyapunov_Exponent'] = df['Log_Divergence'].diff() / np.log(1 + epsilon)
    window_size = 14
    df['Lyapunov_Exponent_MA'] = df['Lyapunov_Exponent'].rolling(window=window_size).mean()

    # Calculate MA and percentage difference
    df['MA_200'] = df['Close'].rolling(window=200, min_periods=200).mean()
    df['Perc_Diff'] = (df['Kalman'] - df['MA_200']) / df['MA_200'] * 100

    # Additional calculations
    df = add_kalman_and_entropy_metrics(df, window_size=70, bins=30)
    df = add_kalman_and_recurrence_metrics(df, epsilon_multiplier=0.01, window_size=70)
    df = add_complexity_metrics(df)
    df = calculate_poc_and_metrics(df, window_size=70)
    df = calculate_percentage_difference_from_ewma(df, 'Close', [14, 151, 269], adjust=False)

    # Mean reversion scores
    windows = [28, 90, 151]
    std_multipliers = [1, 2, 3]
    df = add_multiple_mean_reversion_z_scores(df, 'Smoothed_Close', windows, std_multipliers)

    # Final cleanup
    columns_to_drop = ['Adj Close', 'ATZ_Upper', 'ATZ_Lower', 'VWAP', '200DAY_ATR-', '200DAY_ATR', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = interpolate_columns(df, max_gap_fill=50)
    df = df.iloc[200:]
    df = df.drop(columns=columns_to_drop, axis=1)
    df = df.round(8)

    return df








##===========================(File Processing)===========================##
##===========================(File Processing)===========================##
##===========================(File Processing)===========================##

def clean_and_interpolate_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
    df.ffill(inplace=True)
    return df

def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True


def DataQualityCheck(df, all_dfs=None):
    if df.empty or len(df) < 201:
        logging.error("DataFrame is empty or too short to process.")
        return None

    if df[['Open', 'High', 'Low', 'Close']].std().sum() < 0.01:
        logging.error("Data is flat. Variance in Open, High, Low, Close prices is too low.")
        return None

    if len(df[df['Close'] < 1]) > len(df) / 3:
        logging.error("More than 1/3 of the data has a close price below 1. Skipping the data.")
        return None

    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Add 'Adj Close' if it doesn't exist
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    # **Ensure data goes back to at least 2022**
    earliest_date = df['Date'].min()
    if earliest_date > pd.Timestamp('2022-05-01'):
        logging.error(f"Data does not go back to 2022. Earliest date in dataset: {earliest_date}")
        return None

    # Check if the data is recent (has data in the current year)
    current_year = datetime.now().year
    previous_year = current_year - 1
    if df['Date'].max().year < previous_year:
        logging.error(f"Data is not recent. Last date in dataset: {df['Date'].max()}")
        return None

    # Continue with other checks...
    sample_size = max(int(len(df) * 0.02), 1)
    start_mean = df['Close'].head(sample_size).mean()
    end_mean = df['Close'].tail(sample_size).mean()

    if start_mean > 3000 or (start_mean / max(end_mean, 1e-10) > 20):
        logging.error(f"Signs of reverse stock splits detected or high initial prices: Start mean: {start_mean}, End mean: {end_mean}")
        return None

    return df















def process_file(file_path, output_dir):
    """Process a single parquet file and return success status."""
    try:
        file_path.endswith('.parquet')
        df = pd.read_parquet(file_path)
        if 'Date' not in df.columns:
            if df.index.name == 'Date':
                df = df.reset_index()
            else:
                logging.error("The 'Date' column is missing from the DataFrame.")
                return False

        if df['Date'].dtype != 'datetime64[ns]':
            logging.info("Converting 'Date' column to datetime.")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
        if not validate_columns(df, ['Close', 'High', 'Low', 'Volume']):
            logging.error(f"File {file_path} does not contain all required columns.")
            return False

        df = DataQualityCheck(df)
        if df is None:
            logging.error(f"Data quality check failed for {file_path}.")
            return False

        df = indicators(df)
        df = clean_and_interpolate_data(df)
        SaveData(df, file_path, output_dir)
        return True  # Explicitly return True on success

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        traceback_info = traceback.format_exc()
        logging.error(traceback_info)
        return False
    
    

def SaveData(df, file_path, output_dir):
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, file_name)
    df.to_parquet(output_file, index=False)
    del df

def clear_output_directory(output_dir):
    """
    Remove existing files in the output directory.
    """
    for file in os.listdir(output_dir):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file))

















##===========================(Main Function)===========================##
##===========================(Main Function)===========================##
##===========================(Main Function)===========================##
##===========================(Main Function)===========================##

def process_file_wrapper(file_path):
    return process_file(file_path, CONFIG['output_directory'])


def process_data_files(run_percent):
    print(f"Processing {run_percent}% of files from {CONFIG['input_directory']}")
    StartTimer = time.time()
    
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    clear_output_directory(CONFIG['output_directory'])

    file_paths = [os.path.join(CONFIG['input_directory'], f) 
                 for f in os.listdir(CONFIG['input_directory']) 
                 if f.endswith('.parquet')]
    
    files_to_process = file_paths[:int(len(file_paths) * (run_percent / 100))]
    num_workers = os.cpu_count()
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file_wrapper, file_path) 
                  for file_path in files_to_process]
        
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():  # This will now be True/False
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                finally:
                    pbar.update(1)
                    pbar.set_description(
                        f"Processing files (Success: {completed}, Failed: {failed})"
                    )

    total_time = time.time() - StartTimer
    files_per_second = len(files_to_process) / total_time
    
    print(f"\nProcessed {len(files_to_process)} files in {total_time:.2f} seconds")
    print(f"Files per second: {files_per_second:.2f}")
    print(f'Average time per file: {round(total_time / len(files_to_process), 2)} seconds')
    print(f'Successfully processed: {completed}')
    print(f'Failed to process: {failed}')






if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting the process...")
    parser = argparse.ArgumentParser(description="Process financial market data files.")
    parser.add_argument('--runpercent', type=int, default=100, help="Percentage of files to process from the input directory.")
    args = parser.parse_args()

    process_data_files(args.runpercent)