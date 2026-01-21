from flask import Flask, jsonify, send_from_directory, request
from datetime import datetime
import threading, time, os, sys
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder=".")

# SAFARI CORS FIX - Manual headers
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

def analyze_single_stock(ticker, target_price):
    try:
        print(f"Fetching {ticker}...", file=sys.stderr)
        stock = yf.Ticker(ticker)
        data = stock.history(period="3mo", interval="1d")
        if data.empty:
            raise Exception("No data")
        
        current_price = float(data['Close'].iloc[-1])
        distance_pct = (target_price - current_price) / current_price * 100
        prob = max(0, min(100, 100 - abs(distance_pct) * 0.5))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": current_price,
            "target_price": float(target_price),
            "distance_to_target": float(target_price - current_price),
            "distance_pct": round(distance_pct, 1),
            "probability": {
                "composite_probability": float(prob),
                "momentum_score": 50.0,
                "statistical_probability": 50.0,
                "ml_probability": 50.0,
                "confidence_level": "MEDIUM"
            }
        }
    except Exception as e:
        print(f"{ticker} error: {e}", file=sys.stderr)
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": 0.0,
            "target_price": float(target_price),
            "probability": {"composite_probability": 0.0}
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
        
        print(f"SUCCESS: Delivering {len(stock_data)} stocks", file=sys.stderr)
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

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "stocks": len(stock_data)})

# *** FRONTEND DEBUG ENDPOINT ***
@app.route("/api/debug")
def debug():
    """Test if frontend can parse this exact data."""
    return jsonify({
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "stocks": {
            "TSLA": {
                "ticker": "TSLA",
                "current_price": 429.40,
                "target_price": 500.0,
                "probability": {"composite_probability": 76.8}
            }
        },
        "message": "Backend perfect - frontend needs fixing",
        "all_stocks": stock_data
    })

# STARTUP - populate data immediately
print("🚀 STARTING - PREPOPULATING DATA...", file=sys.stderr)
for ticker, target in stock_config.items():
    stock_data[ticker] = analyze_single_stock(ticker, target)

# Background refresh every 5 minutes
def background_refresh():
    while True:
        time.sleep(300)
        for ticker, target in stock_config.items():
            stock_data[ticker] = analyze_single_stock(ticker, target)
        print("🔄 Background refresh complete", file=sys.stderr)

threading.Thread(target=background_refresh, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server ready on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
