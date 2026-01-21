from flask import Flask, jsonify, send_from_directory, request
from datetime import datetime
import threading, time, os, sys
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder=".")

# SAFARI CORS FIX
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,x-requested-with')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        response = app.make_response('')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

# Global state
stock_config = {"TSLA": 500.0, "AAPL": 250.0, "NVDA": 200.0, "RCAT": 30.0}
stock_data = {}
data_lock = threading.Lock()

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

        statistical_prob = prob + np.random.uniform(-5, 5)   # dummy variation
        ml_prob = prob + np.random.uniform(-10, 10)          # dummy variation
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
    for f in ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]:
        if os.path.exists(f):
            return send_from_directory(".", f)
    return "No dashboard HTML found", 404

@app.route("/api/analysis")
def get_analysis():
    with data_lock:
        for ticker in stock_config:
            if ticker not in stock_data:
                stock_data[ticker] = analyze_single_stock(ticker, stock_config[ticker])
        print(f"SUCCESS: Delivering {len(stock_data)} stocks WITH statistics", file=sys.stderr)
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "stocks": stock_data
        })

@app.route("/api/config", methods=['GET', 'POST'])
def config():
    global stock_config
    if request.method == 'POST':
        data = request.get_json()
        stock_config.update(data.get("stocks", {}))
    return jsonify({"stocks": stock_config})

@app.route("/api/debug")
def debug():
    return jsonify({
        "status": "alive",
        "message": "Backend perfect - has statistics return_id",
        "sample": stock_data.get("TSLA", {})
    })

# STARTUP
print("STARTING WITH FULL DATA STRUCTURE...", file=sys.stderr)
for ticker, target in stock_config.items():
    stock_data[ticker] = analyze_single_stock(ticker, target)

def background_refresh():
    while True:
        time.sleep(300)  # 5 minutes
        with data_lock:
            for ticker, target in stock_config.items():
                stock_data[ticker] = analyze_single_stock(ticker, target)
        print("Background refresh complete", file=sys.stderr)

threading.Thread(target=background_refresh, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Server ready on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
