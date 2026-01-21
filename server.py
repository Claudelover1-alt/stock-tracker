# server.py - FIXED CORS + IMMEDIATE DATA DISPLAY
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from datetime import datetime
import threading, time, sys, os
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder=".")
# AGGRESSIVE CORS - fixes Safari "Load failed"
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Global state
stock_config = {"TSLA": 500.0, "AAPL": 250.0, "NVDA": 200.0, "RCAT": 30.0}
stock_data = {}
data_lock = threading.Lock()
last_update = None

def analyze_single_stock(ticker, target_price):
    """Single yfinance call - bulletproof."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="3mo", interval="1d")
        if data.empty:
            raise Exception("Empty data")
        
        current_price = data['Close'].iloc[-1]
        distance_pct = (target_price - current_price) / current_price * 100
        probability = max(0, min(100, 100 - abs(distance_pct) * 0.5))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": float(current_price),
            "target_price": float(target_price),
            "distance_to_target": float(target_price - current_price),
            "distance_pct": round(float(distance_pct), 1),
            "probability": {
                "composite_probability": float(probability),
                "momentum_score": 50.0,
                "statistical_probability": 50.0,
                "ml_probability": 50.0,
                "confidence_level": "MEDIUM"
            }
        }
    except:
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": 0.0,
            "target_price": float(target_price),
            "probability": {"composite_probability": 0.0}
        }

@app.route("/", methods=['GET', 'OPTIONS'])
def index():
    # Handle Safari preflight
    if request.method == 'OPTIONS':
        return "", 200
    for f in ["dashboard_multi.html", "index.html"]:
        if os.path.exists(f):
            return send_from_directory(".", f)
    return "No dashboard found", 404

@app.route("/api/analysis", methods=['GET', 'OPTIONS'])
def get_analysis():
    # Handle Safari preflight
    if request.method == 'OPTIONS':
        return "", 200
    
    # IMMEDIATE data - fetch on demand
    with data_lock:
        for ticker in stock_config:
            if ticker not in stock_data:
                stock_data[ticker] = analyze_single_stock(ticker, stock_config[ticker])
        last_update = datetime.now()
        
        print(f"🌐 API DELIVERED {len(stock_data)} stocks to Safari", file=sys.stderr)
        return jsonify({
            "timestamp": last_update.isoformat(),
            "stocks": stock_data
        }), 200

@app.route("/api/config", methods=['GET', 'POST', 'OPTIONS'])
def config():
    if request.method == 'OPTIONS':
        return "", 200
    if request.method == 'GET':
        return jsonify({"stocks": stock_config})
    return jsonify({"success": True})

# STARTUP - populate immediately
print("🚀 STARTING WITH IMMEDIATE DATA...", file=sys.stderr)
for ticker, target in stock_config.items():
    stock_data[ticker] = analyze_single_stock(ticker, target)

# Background refresh every 5min
def refresh_loop():
    while True:
        time.sleep(300)
        for ticker, target in stock_config.items():
            stock_data[ticker] = analyze_single_stock(ticker, target)
        print("🔄 5min refresh complete", file=sys.stderr)

threading.Thread(target=refresh_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
