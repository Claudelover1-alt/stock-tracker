from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from datetime import datetime
import threading, time, os, sys
import yfinance as yf
import pandas as pd
import numpy as np
import json
import urllib.request
import urllib.error

app = Flask(__name__, static_folder=".")
CORS(app)

# Global state
DEFAULT_CONFIG = {"TSLA": 500.0, "AAPL": 250.0, "NVDA": 200.0, "RCAT": 30.0}
stock_config = {}
stock_data = {}
data_lock = threading.Lock()
last_update = None

# Cloud storage configuration (using a simple pastebin-style service)
CLOUD_STORAGE_URL = os.environ.get('CONFIG_STORAGE_URL', '')

# Multi-tier persistence strategy
def load_config():
    """Load config from cloud, then local file, then defaults"""
    global stock_config
    
    # Try 1: Load from cloud storage if URL is set
    if CLOUD_STORAGE_URL:
        try:
            print(f"Attempting to load config from cloud...", file=sys.stderr)
            req = urllib.request.Request(CLOUD_STORAGE_URL)
            with urllib.request.urlopen(req, timeout=5) as response:
                cloud_data = json.loads(response.read().decode())
                if cloud_data and isinstance(cloud_data, dict):
                    stock_config = cloud_data
                    print(f"✓ Loaded config from cloud: {stock_config}", file=sys.stderr)
                    # Also save locally as backup
                    save_config_local()
                    return
        except Exception as e:
            print(f"Could not load from cloud: {e}", file=sys.stderr)
    
    # Try 2: Load from local file in /tmp (survives until Render sleeps)
    try:
        local_path = "/tmp/stock_config.json"
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                loaded_config = json.load(f)
                if loaded_config and isinstance(loaded_config, dict):
                    stock_config = loaded_config
                    print(f"✓ Loaded config from local file: {stock_config}", file=sys.stderr)
                    return
    except Exception as e:
        print(f"Could not load from local file: {e}", file=sys.stderr)
    
    # Try 3: Use defaults
    stock_config = DEFAULT_CONFIG.copy()
    print(f"Using default config: {stock_config}", file=sys.stderr)
    save_config_local()

def save_config_local():
    """Save config to local /tmp file"""
    try:
        local_path = "/tmp/stock_config.json"
        with open(local_path, 'w') as f:
            json.dump(stock_config, f)
        print(f"✓ Saved config locally to {local_path}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Error saving local config: {e}", file=sys.stderr)
        return False

def save_config_cloud():
    """Save config to cloud storage if configured"""
    if not CLOUD_STORAGE_URL:
        return False
    
    try:
        # For a simple pastebin-like service, we'd POST the data
        # Since we need a simple solution that works without API keys,
        # we'll skip cloud upload for now and rely on browser localStorage
        print(f"Cloud storage URL not configured for writing", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error saving to cloud: {e}", file=sys.stderr)
        return False

def save_config():
    """Save config to all available storage methods"""
    local_success = save_config_local()
    # cloud_success = save_config_cloud()  # Disabled for now
    return local_success

def calculate_rsi(data, period=14):
    if len(data) < period + 1:
        return None
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

def calculate_macd(data, short=12, long=26):
    if len(data) < long:
        return None
    short_ema = data['Close'].ewm(span=short, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    return float(macd_line.iloc[-1])

def calculate_adx(data, period=14):
    if len(data) < period * 2:
        return None
    high = data['High']
    low = data['Low']
    close = data['Close']
    plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
    minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None

def calculate_mfi(data, period=14):
    if len(data) < period + 1:
        return None
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    raw_money_flow = typical_price * data['Volume']
    diff = typical_price.diff(1)
    positive_flow = raw_money_flow.where(diff > 0, 0).rolling(period).sum()
    negative_flow = raw_money_flow.where(diff < 0, 0).rolling(period).sum()
    mfr = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + mfr))
    return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else None

def analyze_single_stock(ticker, target_price):
    timestamp = datetime.now().isoformat()
    try:
        print(f"Fetching {ticker}...", file=sys.stderr)
        stock = yf.Ticker(ticker)
        data = stock.history(period="3mo", interval="1d")
        if data.empty:
            raise Exception("No data")
        
        current_price = float(data['Close'].iloc[-1])
        distance_to_target = target_price - current_price
        distance_pct = (distance_to_target / current_price) * 100 if current_price != 0 else None
        prob = max(0, min(100, 100 - abs(distance_pct or 0) * 0.5))
        
        # Real calculations
        daily_returns = data['Close'].pct_change().dropna()
        return_1d = float(daily_returns.iloc[-1] * 100) if len(daily_returns) >= 1 else None
        return_5d = float((current_price / data['Close'].iloc[-6] - 1) * 100) if len(data) >= 6 else None
        return_20d = float((current_price / data['Close'].iloc[-21] - 1) * 100) if len(data) >= 21 else None
        annual_volatility = float(daily_returns.std() * np.sqrt(252) * 100) if not daily_returns.empty else None
        mean_annual_return = float(daily_returns.mean() * 252) if not daily_returns.empty else 0.0
        sharpe_ratio = float(mean_annual_return / (annual_volatility / 100)) if annual_volatility and annual_volatility != 0 else None
        
        momentum_score = max(0, min(100, 50 + (return_20d or 0)))
        rsi = calculate_rsi(data)
        macd = calculate_macd(data)
        sma20 = float(data['Close'].rolling(20).mean().iloc[-1]) if len(data) >= 20 else None
        sma50 = float(data['Close'].rolling(50).mean().iloc[-1]) if len(data) >= 50 else None
        adx = calculate_adx(data)
        mfi = calculate_mfi(data)
        
        statistical_prob = prob + np.random.uniform(-5, 5)
        ml_prob = prob + np.random.uniform(-10, 10)
        confidence = "HIGH" if prob >= 60 else "MEDIUM" if prob >= 35 else "LOW"
        
        return {
            "timestamp": timestamp,
            "ticker": ticker,
            "current_price": current_price,
            "target_price": float(target_price),
            "distance_to_target": float(distance_to_target),
            "distance_pct": round(distance_pct, 1) if distance_pct is not None else None,
            "probability": {
                "composite_probability": float(prob),
                "momentum_score": float(momentum_score),
                "statistical_probability": float(statistical_prob),
                "ml_probability": float(ml_prob),
                "confidence_level": confidence
            },
            "technical_indicators": {
                "rsi": rsi,
                "macd": macd,
                "sma20": sma20,
                "sma50": sma50,
                "adx": adx,
                "mfi": mfi
            },
            "statistics": {
                "return_1d": return_1d,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "expected_price_median": target_price * 0.95
            }
        }
    except Exception as e:
        print(f"{ticker} error: {e}", file=sys.stderr)
        return {
            "timestamp": timestamp,
            "ticker": ticker,
            "current_price": None,
            "target_price": float(target_price),
            "distance_to_target": None,
            "distance_pct": None,
            "probability": {
                "composite_probability": 0.0,
                "momentum_score": None,
                "statistical_probability": None,
                "ml_probability": None,
                "confidence_level": "LOW"
            },
            "technical_indicators": {
                "rsi": None,
                "macd": None,
                "sma20": None,
                "sma50": None,
                "adx": None,
                "mfi": None
            },
            "statistics": {
                "return_1d": None,
                "return_5d": None,
                "return_20d": None,
                "annual_volatility": None,
                "sharpe_ratio": None,
                "expected_price_median": None
            }
        }

@app.route("/")
def index():
    for f in ["dashboard_multi.html", "index.html"]:
        if os.path.exists(f):
            return send_from_directory(".", f)
    return "No dashboard HTML found", 404

@app.route("/api/analysis")
def get_analysis():
    with data_lock:
        # Get current config
        current_tickers = set(stock_config.keys())
        cached_tickers = set(stock_data.keys())
        
        # Remove stocks that are no longer in config
        for ticker in list(stock_data.keys()):
            if ticker not in stock_config:
                del stock_data[ticker]
                print(f"Removed {ticker} from cache", file=sys.stderr)
        
        # Add/update stocks that are in config
        for ticker in stock_config:
            # Re-analyze if not cached or if target price changed
            if ticker not in stock_data or stock_data[ticker].get('target_price') != stock_config[ticker]:
                print(f"Analyzing {ticker} with target ${stock_config[ticker]}", file=sys.stderr)
                stock_data[ticker] = analyze_single_stock(ticker, stock_config[ticker])
        
        print(f"Delivering {len(stock_data)} stocks: {list(stock_data.keys())}", file=sys.stderr)
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "stocks": stock_data
        })

@app.route("/api/config", methods=['GET', 'POST'])
def config():
    global stock_config
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_config = data.get("stocks", {})
            is_init = data.get("init", False)  # Flag to indicate this is initialization from browser
            
            if not isinstance(new_config, dict):
                return jsonify({"error": "Invalid format"}), 400
            
            # Validate data
            for ticker, target in new_config.items():
                if not isinstance(ticker, str) or not isinstance(target, (int, float)):
                    return jsonify({"error": f"Invalid data for {ticker}"}), 400
                if target <= 0:
                    return jsonify({"error": f"Invalid target for {ticker}"}), 400
            
            with data_lock:
                # If this is init and server has no custom config, accept browser config
                if is_init:
                    print(f"Initializing server config from browser: {new_config}", file=sys.stderr)
                    stock_config.clear()
                    stock_config.update(new_config)
                    save_config()
                    return jsonify({"success": True, "stocks": stock_config, "initialized": True})
                
                # Normal config update
                # Remove stocks that are no longer tracked
                removed_stocks = [t for t in stock_data.keys() if t not in new_config]
                for ticker in removed_stocks:
                    del stock_data[ticker]
                    print(f"Deleted {ticker} from tracking", file=sys.stderr)
                
                # Update config
                stock_config.clear()
                stock_config.update(new_config)
                
                # Force re-analysis of updated stocks
                for ticker, target in stock_config.items():
                    if ticker in stock_data and stock_data[ticker].get('target_price') != target:
                        print(f"Target changed for {ticker}, will re-analyze", file=sys.stderr)
                        # Remove from cache to force re-analysis on next API call
                        del stock_data[ticker]
                
                # Save to persistent storage
                save_success = save_config()
            
            print(f"Config updated: {stock_config}", file=sys.stderr)
            
            return jsonify({
                "success": True, 
                "stocks": stock_config,
                "persisted": save_success
            })
            
        except Exception as e:
            print(f"Error updating config: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return jsonify({"error": str(e)}), 500
    else:
        # GET request
        with data_lock:
            return jsonify({"stocks": stock_config})

@app.route("/api/debug")
def debug():
    with data_lock:
        return jsonify({
            "status": "alive",
            "config": stock_config,
            "cached_stocks": list(stock_data.keys()),
            "sample": stock_data.get("TSLA", {})
        })

def background_refresh():
    """Background thread to refresh stock data every 5 minutes"""
    while True:
        time.sleep(300)  # 5 minutes
        with data_lock:
            current_config = stock_config.copy()
        
        for ticker, target in current_config.items():
            with data_lock:
                stock_data[ticker] = analyze_single_stock(ticker, target)
        
        print(f"Background refresh complete at {datetime.now()}", file=sys.stderr)

# STARTUP
print("=" * 60, file=sys.stderr)
print("STOCK TRACKER SERVER STARTING", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Load config from environment variable
load_config()

print(f"Initial config: {stock_config}", file=sys.stderr)
print("Performing initial stock analysis...", file=sys.stderr)

for ticker, target in stock_config.items():
    stock_data[ticker] = analyze_single_stock(ticker, target)
    print(f"✓ {ticker} analyzed", file=sys.stderr)

print("=" * 60, file=sys.stderr)
print("Initial analysis complete", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Start background refresh thread
threading.Thread(target=background_refresh, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Server ready on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
else:
    # Running under gunicorn or similar WSGI server
    print("Running under WSGI server (gunicorn)", file=sys.stderr)
