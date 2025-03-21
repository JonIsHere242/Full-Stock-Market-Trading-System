import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_from_directory
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask_socketio import SocketIO, emit
import threading
import logging
import traceback


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dashboard.log'
)

app = Flask(__name__, 
            template_folder='app/templates',
            static_folder='app/static')
app.config['SECRET_KEY'] = 'stocksniper-dashboard-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Directory paths - adjust these to match your file structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

# Parquet file paths - CRITICAL FOR OPERATION
BUY_SIGNALS_PATH = os.path.join(BASE_DIR, "_Buy_Signals.parquet")
LIVE_TRADES_PATH = os.path.join(BASE_DIR, "_Live_trades.parquet")

# Cache for performance
data_cache = {
    'last_update': None,
    'buy_signals': None,
    'live_trades': None,
    'metrics': None,
    'charts': {}
}

# Cache timeout in seconds
CACHE_TIMEOUT = 10

# Sample historical price data (for charts when market is closed)
HISTORICAL_DATA = {}



def load_buy_signals(force_refresh=False):
    """Load current buy signals from parquet file with correct typing"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        data_cache['buy_signals'] is not None and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['buy_signals']
    
    try:
        if not os.path.exists(BUY_SIGNALS_PATH):
            logging.warning(f"Buy signals file not found at: {BUY_SIGNALS_PATH}")
            return pd.DataFrame(columns=[
                'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice',
                'IsCurrentlyBought', 'ConsecutiveLosses', 'LastTradedDate',
                'UpProbability', 'LastSellPrice', 'PositionSize'
            ])
        
        # Load data
        df = pd.read_parquet(BUY_SIGNALS_PATH)
        
        # Ensure column types
        if 'Symbol' in df.columns and not pd.api.types.is_string_dtype(df['Symbol']):
            df['Symbol'] = df['Symbol'].astype(str)
        
        if 'IsCurrentlyBought' in df.columns:
            df['IsCurrentlyBought'] = df['IsCurrentlyBought'].astype(bool)
            
        for col in ['LastBuySignalPrice', 'UpProbability', 'LastSellPrice', 'PositionSize']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        for col in ['LastBuySignalDate', 'LastTradedDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Log info
        logging.info(f"Loaded buy signals file with {len(df)} rows")
        if not df.empty:
            logging.info(f"First row: {df.iloc[0].to_dict()}")
            
        if 'IsCurrentlyBought' in df.columns:
            active_count = df['IsCurrentlyBought'].sum()
            logging.info(f"Active positions count: {active_count}")
        
        # Update cache
        data_cache['buy_signals'] = df
        data_cache['last_update'] = current_time
        
        return df
    except Exception as e:
        logging.error(f"Error loading buy signals: {str(e)}")
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice',
            'IsCurrentlyBought', 'ConsecutiveLosses', 'LastTradedDate',
            'UpProbability', 'LastSellPrice', 'PositionSize'
        ])



def load_live_trades(force_refresh=False):
    """Load live trades history from parquet file with caching"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        data_cache['live_trades'] is not None and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['live_trades']
    
    try:
        # Check if file exists
        if not os.path.exists(LIVE_TRADES_PATH):
            logging.warning(f"Live trades file not found at: {LIVE_TRADES_PATH}")
            return pd.DataFrame(columns=[
                'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice',
                'IsCurrentlyBought', 'ConsecutiveLosses', 'LastTradedDate',
                'UpProbability', 'LastSellPrice', 'PositionSize'
            ])
        
        # Load and process the data
        df = pd.read_parquet(LIVE_TRADES_PATH)
        
        # Handle datetime columns for JSON serialization
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update cache
        data_cache['live_trades'] = df
        data_cache['last_update'] = current_time
        
        logging.info(f"Loaded {len(df)} live trades from {LIVE_TRADES_PATH}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading live trades: {str(e)}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice',
            'IsCurrentlyBought', 'ConsecutiveLosses', 'LastTradedDate',
            'UpProbability', 'LastSellPrice', 'PositionSize'
        ])



def load_historical_price_data(symbol, days=100):
    """
    Load historical price data for a symbol from the PriceData directory
    """
    # Path to your price data file
    price_data_path = os.path.join(DATA_DIR, 'PriceData', f'{symbol}.parquet')
    
    try:
        if os.path.exists(price_data_path):
            # Load actual historical price data
            df = pd.read_parquet(price_data_path)
            
            # Convert Date column to datetime if needed
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date and limit to last N days
            df = df.sort_values('Date')
            if len(df) > days:
                df = df.tail(days)
            
            logging.info(f"Loaded {len(df)} days of price data for {symbol}")
            return df
        else:
            logging.warning(f"Price data file not found: {price_data_path}")
            return generate_mock_price_data(symbol, days)
    except Exception as e:
        logging.error(f"Error loading price data for {symbol}: {str(e)}")
        return generate_mock_price_data(symbol, days)




def generate_mock_price_data(symbol, days=100):
    """Generate mock price data as fallback when actual data cannot be loaded"""
    logging.warning(f"Generating mock data for {symbol}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Start price between $10-200
    start_price = np.random.uniform(10, 200)
    
    # Generate prices with random walk
    prices = [start_price]
    for i in range(1, len(date_range)):
        change = np.random.normal(0.001, 0.02)  
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Open': prices,
        'High': [p * np.random.uniform(1.001, 1.02) for p in prices],
        'Low': [p * np.random.uniform(0.98, 0.999) for p in prices],
        'Close': [p * np.random.uniform(0.995, 1.005) for p in prices],
        'Volume': np.random.randint(100000, 10000000, len(date_range)),
        'Ticker': symbol
    })
    
    HISTORICAL_DATA[symbol] = df
    return df



def calculate_pnl(force_refresh=False):
    """Calculate P&L metrics from actual trade history"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        data_cache['metrics'] is not None and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['metrics']
    
    try:
        # Load buy signals for active positions
        active_df = load_buy_signals(force_refresh)
        
        # Load trade history for completed trades
        trade_history_path = os.path.join(BASE_DIR, "trade_history.parquet")
        
        if not os.path.exists(trade_history_path):
            logging.warning(f"Trade history file not found at: {trade_history_path}")
            return default_metrics(active_df)
            
        trades_df = pd.read_parquet(trade_history_path)
        
        if len(trades_df) == 0:
            return default_metrics(active_df)
        
        # Calculate metrics from trade history
        total_pnl = trades_df['PnL'].sum()
        
        # Win/loss analysis
        winning_trades = trades_df[trades_df['PnL'] > 0]
        losing_trades = trades_df[trades_df['PnL'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
        
        # Count active positions
        active_positions = 0
        if 'IsCurrentlyBought' in active_df.columns:
            # Convert to boolean if it's not already
            if active_df['IsCurrentlyBought'].dtype != bool:
                active_df['IsCurrentlyBought'] = active_df['IsCurrentlyBought'].astype(str).str.lower() == 'true'
            active_positions = len(active_df[active_df['IsCurrentlyBought'] == True])
        
        metrics = {
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2) if avg_loss < 0 else 0,
            'total_trades': len(trades_df),
            'active_positions': active_positions
        }
        
        # Update cache
        data_cache['metrics'] = metrics
        data_cache['last_update'] = current_time
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error calculating P&L: {str(e)}")
        return default_metrics()

def default_metrics(active_df=None):
    """Return default metrics when no trade data is available"""
    active_positions = 0
    if active_df is not None and 'IsCurrentlyBought' in active_df.columns:
        # Convert to boolean if it's not already
        if active_df['IsCurrentlyBought'].dtype != bool:
            active_df['IsCurrentlyBought'] = active_df['IsCurrentlyBought'].astype(str).str.lower() == 'true'
        active_positions = len(active_df[active_df['IsCurrentlyBought'] == True])
    
    return {
        'total_pnl': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'total_trades': 0,
        'active_positions': active_positions
    }





def create_price_chart(symbol, force_refresh=False):
    """Create a price chart for a specific symbol with caching"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        symbol in data_cache['charts'] and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['charts'][symbol]
    
    try:
        df = load_historical_price_data(symbol)
        
        if df.empty:
            return create_empty_chart(f"No price data available for {symbol}")
        
        # Create a candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol
        )])
        
        # Add volume as a bar chart on a secondary y-axis
        fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker=dict(color='rgba(0, 0, 255, 0.3)'),
            opacity=0.3,
            yaxis='y2'
        ))
        
        # Add buy and sell markers if available in trade history
        try:
            # Check for buy signals
            buy_signals = load_buy_signals(force_refresh=True)
            
            # Check for this symbol
            if 'Symbol' in buy_signals.columns and symbol in buy_signals['Symbol'].values:
                signal_row = buy_signals[buy_signals['Symbol'] == symbol].iloc[0]
                if pd.notna(signal_row.get('LastBuySignalDate')):
                    buy_date = pd.to_datetime(signal_row['LastBuySignalDate'])
                    buy_price = signal_row.get('LastBuySignalPrice', 0)
                    
                    # Add buy marker
                    fig.add_trace(go.Scatter(
                        x=[buy_date],
                        y=[buy_price],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        )
                    ))
            
            # Check for sell signals (from trade history)
            trade_history_path = os.path.join(BASE_DIR, "trade_history.parquet")
            if os.path.exists(trade_history_path):
                trades_df = pd.read_parquet(trade_history_path)
                
                # Filter for this symbol
                if 'Symbol' in trades_df.columns and symbol in trades_df['Symbol'].values:
                    symbol_trades = trades_df[trades_df['Symbol'] == symbol]
                    
                    # Add sell markers for each exit
                    for _, trade in symbol_trades.iterrows():
                        if pd.notna(trade.get('ExitDate')) and pd.notna(trade.get('ExitPrice')):
                            exit_date = pd.to_datetime(trade['ExitDate'])
                            exit_price = trade['ExitPrice']
                            
                            # Add sell marker
                            fig.add_trace(go.Scatter(
                                x=[exit_date],
                                y=[exit_price],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=12,
                                    color='red',
                                    line=dict(width=2, color='darkred')
                                )
                            ))
        except Exception as e:
            logging.error(f"Error adding trade markers: {str(e)}")
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Price History',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_rangeslider_visible=False,
            height=400,
            template='plotly_dark',
            margin=dict(l=50, r=50, t=50, b=30)
        )
        
        chart_json = json.loads(plotly.io.to_json(fig))
        
        # Update cache
        data_cache['charts'][symbol] = chart_json
        
        return chart_json
    
    except Exception as e:
        logging.error(f"Error creating price chart for {symbol}: {str(e)}")
        logging.error(traceback.format_exc())
        return create_empty_chart(f"Error loading chart for {symbol}: {str(e)}")

def create_empty_chart(message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        template='plotly_dark',
        height=400
    )
    return json.loads(plotly.io.to_json(fig))


def create_performance_chart(force_refresh=False):
    """Create a performance chart based on actual trade history"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        'performance' in data_cache['charts'] and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['charts']['performance']
    
    try:
        # Load trade history from parquet file
        trade_history_path = os.path.join(BASE_DIR, "trade_history.parquet")
        
        if not os.path.exists(trade_history_path):
            logging.warning(f"Trade history file not found at: {trade_history_path}")
            return create_empty_performance_chart("No trade history found")
            
        trades_df = pd.read_parquet(trade_history_path)
        
        if len(trades_df) == 0:
            return create_empty_performance_chart("No trades in history")
            
        # Ensure date columns are datetime
        for col in ['EntryDate', 'ExitDate']:
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce')
        
        # Sort by exit date
        trades_df = trades_df.sort_values('ExitDate')
        
        # Create cumulative P&L
        trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
        
        # Create the chart
        fig = go.Figure()
        
        # Add cumulative P&L line
        fig.add_trace(go.Scatter(
            x=trades_df['ExitDate'],
            y=trades_df['Cumulative_PnL'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='green', width=2)
        ))
        
        # Add individual trade markers
        fig.add_trace(go.Scatter(
            x=trades_df['ExitDate'],
            y=trades_df['PnL'],
            mode='markers',
            name='Individual Trades',
            marker=dict(
                color=trades_df['PnL'].apply(lambda x: 'green' if x > 0 else 'red'),
                size=8
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Trading Performance',
            xaxis_title='Date',
            yaxis_title='P&L ($)',
            template='plotly_dark',
            height=320,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=50, b=30)
        )
        
        chart_json = json.loads(plotly.io.to_json(fig))
        
        # Update cache
        data_cache['charts']['performance'] = chart_json
        
        return chart_json
    
    except Exception as e:
        logging.error(f"Error creating performance chart: {str(e)}")
        return create_empty_performance_chart(f"Error: {str(e)}")

def create_empty_performance_chart(message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        template='plotly_dark',
        height=320
    )
    return json.loads(plotly.io.to_json(fig))


def create_up_probability_distribution(force_refresh=False):
    """Create a histogram of actual UpProbability values from signals"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        'probability' in data_cache['charts'] and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['charts']['probability']
    
    try:
        # Load both current signals and trade history
        signals_df = load_buy_signals(force_refresh)
        
        # Also check trade history for more data points
        trade_history_path = os.path.join(BASE_DIR, "trade_history.parquet")
        if os.path.exists(trade_history_path):
            trades_df = pd.read_parquet(trade_history_path)
            
            # Combine UpProbability values from both sources
            up_probs = []
            
            if 'UpProbability' in signals_df.columns:
                up_probs.extend(signals_df['UpProbability'].dropna().tolist())
                
            if 'UpProbability' in trades_df.columns:
                up_probs.extend(trades_df['UpProbability'].dropna().tolist())
        else:
            # Just use signals data
            up_probs = signals_df['UpProbability'].dropna().tolist() if 'UpProbability' in signals_df.columns else []
        
        if not up_probs:
            return create_empty_chart("No UpProbability data available")
            
        # Create histogram data
        hist_data = pd.DataFrame({'UpProbability': up_probs})
        
        # Create histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=hist_data['UpProbability'],
            nbinsx=20,
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add vertical line for average
        avg_prob = np.mean(up_probs)
        fig.add_vline(
            x=avg_prob,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Avg: {avg_prob:.2f}",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title='Distribution of UpProbability Values',
            xaxis_title='UpProbability',
            yaxis_title='Count',
            template='plotly_dark',
            height=320,
            margin=dict(l=40, r=40, t=50, b=30)
        )
        
        chart_json = json.loads(plotly.io.to_json(fig))
        
        # Update cache
        data_cache['charts']['probability'] = chart_json
        
        return chart_json
    
    except Exception as e:
        logging.error(f"Error creating UpProbability distribution: {str(e)}")
        return create_empty_chart(f"Error: {str(e)}")

def create_empty_chart(message):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        template='plotly_dark',
        height=320
    )
    return json.loads(plotly.io.to_json(fig))

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/active_trades')
def active_trades():
    """API endpoint to get active trades data"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        df = load_buy_signals(force_refresh)
        
        # Log the loaded dataframe information
        logging.info(f"Loaded buy signals: {len(df)} rows, columns: {df.columns.tolist()}")
        
        if df.empty:
            logging.warning("Buy signals dataframe is empty")
            return jsonify([])
            
        # Convert IsCurrentlyBought to boolean if it's not already
        if 'IsCurrentlyBought' in df.columns:
            if df['IsCurrentlyBought'].dtype != bool:
                df['IsCurrentlyBought'] = df['IsCurrentlyBought'].astype(str).str.lower() == 'true'
            
            # Log how many active positions we have
            active_count = len(df[df['IsCurrentlyBought'] == True])
            logging.info(f"Found {active_count} active positions out of {len(df)} signals")
            
            # Get active positions
            active_df = df[df['IsCurrentlyBought'] == True].copy()
            
            # Properly handle NaN values and date formatting for JSON serialization
            for col in active_df.columns:
                if pd.api.types.is_datetime64_any_dtype(active_df[col]):
                    active_df[col] = active_df[col].dt.strftime('%Y-%m-%d')
                elif pd.api.types.is_float_dtype(active_df[col]):
                    # Replace NaN with None for JSON serialization
                    active_df[col] = active_df[col].replace({np.nan: None})
            
            # Convert to records
            active = active_df.to_dict(orient='records')
            logging.info(f"Returning {len(active)} active positions")
            return jsonify(active)
        else:
            logging.warning("IsCurrentlyBought column not found in buy signals dataframe")
            return jsonify([])
    except Exception as e:
        logging.error(f"Error in active_trades API: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500






@app.route('/api/available_symbols')
def available_symbols():
    """API endpoint to get all available symbols from price data directory"""
    try:
        price_data_dir = os.path.join(DATA_DIR, 'PriceData')
        if not os.path.exists(price_data_dir):
            logging.warning(f"Price data directory not found: {price_data_dir}")
            return jsonify([])
            
        # Get all parquet files in the directory
        symbols = []
        for file in os.listdir(price_data_dir):
            if file.endswith('.parquet'):
                symbol = file.replace('.parquet', '')
                symbols.append(symbol)
                
        # Sort alphabetically
        symbols.sort()
        
        logging.info(f"Found {len(symbols)} available symbols")
        return jsonify(symbols)
    except Exception as e:
        logging.error(f"Error getting available symbols: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify([]), 500

def load_trade_history(force_refresh=False):
    """Load trade history from the trade_history.parquet file"""
    current_time = time.time()
    
    # Return cached data if available and not expired
    if (not force_refresh and 
        'trade_history' in data_cache and 
        data_cache['last_update'] is not None and 
        current_time - data_cache['last_update'] < CACHE_TIMEOUT):
        return data_cache['trade_history']
    
    try:
        # Path to trade history file
        trade_history_path = os.path.join(BASE_DIR, "trade_history.parquet")
        
        if not os.path.exists(trade_history_path):
            logging.warning(f"Trade history file not found at: {trade_history_path}")
            return pd.DataFrame(columns=[
                'Symbol', 'EntryDate', 'ExitDate', 'EntryPrice', 'ExitPrice', 
                'Quantity', 'PnL', 'PnLPct', 'Commission'
            ])
        
        # Load the data
        df = pd.read_parquet(trade_history_path)
        
        # Handle datetime columns for JSON serialization
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        # Update cache
        data_cache['trade_history'] = df
        data_cache['last_update'] = current_time
        
        logging.info(f"Loaded {len(df)} trade history records from {trade_history_path}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading trade history: {str(e)}")
        logging.error(traceback.format_exc())
        return pd.DataFrame(columns=[
            'Symbol', 'EntryDate', 'ExitDate', 'EntryPrice', 'ExitPrice', 
            'Quantity', 'PnL', 'PnLPct', 'Commission'
        ])

@app.route('/api/trade_history')
def trade_history():
    """API endpoint to get trade history data"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        df = load_trade_history(force_refresh)
        
        if df.empty:
            return jsonify([])
            
        # Handle NaN values for JSON serialization
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].replace({np.nan: None})
            
        history = df.to_dict(orient='records')
        return jsonify(history)
    except Exception as e:
        logging.error(f"Error in trade_history API: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    



@app.route('/api/metrics')
def metrics():
    """API endpoint to get performance metrics"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        metrics = calculate_pnl(force_refresh=force_refresh)
        return jsonify(metrics)
    except Exception as e:
        logging.error(f"Error in metrics API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/charts/price/<symbol>')
def price_chart(symbol):
    """API endpoint to get price chart for a symbol"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        chart_data = create_price_chart(symbol, force_refresh)
        return jsonify(chart_data)
    except Exception as e:
        logging.error(f"Error in price_chart API for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/charts/performance')
def performance_chart():
    """API endpoint to get performance chart"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        chart_data = create_performance_chart(force_refresh=force_refresh)
        return jsonify(chart_data)
    except Exception as e:
        logging.error(f"Error in performance_chart API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/charts/probability')
def probability_chart():
    """API endpoint to get UpProbability distribution chart"""
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        chart_data = create_up_probability_distribution(force_refresh=force_refresh)
        return jsonify(chart_data)
    except Exception as e:
        logging.error(f"Error in probability_chart API: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/kill_position', methods=['POST'])
def kill_position():
    """API endpoint to kill a position"""
    try:
        data = request.json
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        # In a real system, this would trigger your position closing logic
        # For now, we'll just simulate it by updating the parquet file
        
        # Load current signals
        signals_df = load_buy_signals(force_refresh=True)
        
        # Convert to proper types if needed
        if 'Symbol' in signals_df.columns and signals_df['Symbol'].dtype != 'object':
            signals_df['Symbol'] = signals_df['Symbol'].astype(str)
            
        if 'IsCurrentlyBought' in signals_df.columns and signals_df['IsCurrentlyBought'].dtype != bool:
            signals_df['IsCurrentlyBought'] = signals_df['IsCurrentlyBought'].astype(str).str.lower() == 'true'
        
        # Check if the symbol exists and is currently bought
        if symbol not in signals_df['Symbol'].values:
            return jsonify({"error": f"Symbol {symbol} not found"}), 404
        
        mask = (signals_df['Symbol'] == symbol) & (signals_df['IsCurrentlyBought'] == True)
        if not mask.any():
            return jsonify({"error": f"Symbol {symbol} is not currently bought"}), 400
        
        # Update the signal (in a real system, this would need to be synchronized with your actual trade execution)
        signals_df.loc[mask, 'IsCurrentlyBought'] = False
        signals_df.loc[mask, 'LastTradedDate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Here we would save the updated DataFrame back to the parquet file
        # In a production system, you would need proper file locking
        # signals_df.to_parquet(BUY_SIGNALS_PATH)
        
        # For this demo, we'll just log the action
        logging.info(f"Emergency exit triggered for {symbol}")
        
        # Clear cache
        data_cache['buy_signals'] = None
        data_cache['last_update'] = None
        
        # Emit a socket event to update the UI
        socketio.emit('position_killed', {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return jsonify({
            "success": True,
            "message": f"Emergency exit triggered for {symbol}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Error in kill_position API: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Automatic background data refresh (every CACHE_TIMEOUT seconds)
def background_refresh():
    """Background thread to refresh data periodically"""
    while True:
        try:
            logging.info("Running background data refresh")
            # Force refresh all data
            load_buy_signals(force_refresh=True)
            load_live_trades(force_refresh=True)
            calculate_pnl(force_refresh=True)
            
            # Also refresh charts for active symbols
            signals_df = load_buy_signals()
            if 'IsCurrentlyBought' in signals_df.columns:
                if signals_df['IsCurrentlyBought'].dtype != bool:
                    signals_df['IsCurrentlyBought'] = signals_df['IsCurrentlyBought'].astype(str).str.lower() == 'true'
                
                active_symbols = signals_df[signals_df['IsCurrentlyBought'] == True]['Symbol'].tolist()
                for symbol in active_symbols:
                    create_price_chart(symbol, force_refresh=True)
            
            create_performance_chart(force_refresh=True)
            create_up_probability_distribution(force_refresh=True)
            
            # Emit refresh event to clients
            socketio.emit('data_refreshed', {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            logging.error(f"Error in background refresh: {str(e)}")
        
        # Sleep for the cache timeout period
        time.sleep(CACHE_TIMEOUT)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logging.info(f"Client connected: {request.sid}")
    emit('connection_success', {'data': 'Connection established'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logging.info(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Print startup information
    print(f"Dashboard starting on http://localhost:5000")
    print(f"Looking for buy signals at: {BUY_SIGNALS_PATH}")
    print(f"Looking for live trades at: {LIVE_TRADES_PATH}")
    
    # Start background refresh thread
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()
    
    # Start the SocketIO server
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)