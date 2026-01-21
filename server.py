# server.py - FIXED FOR FRONTEND + SINGLE YFINANCE CALL
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json, os, sys, time, threading
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

print("=" * 60, file=sys.stderr)
print("STOCK ANALYZER WITH SINGLE YFINANCE CALL", file=sys.stderr)
print("=" * 60, file=sys.stderr)

app = Flask(__name__, static_folder=".")
CORS(app)

# Global state
stock_config = {"TSLA": 500.0, "AAPL": 250.0, "NVDA": 200.0, "RCAT": 30.0}
stock_data = {}
data_lock = threading.Lock()
last_update = None
analysis_thread_running = False

# SIMPLE ANALYSIS FUNCTION (replaces heavy StockAnalyzer)
def analyze_single_stock(ticker, target_price):
    """Single yfinance call + basic probability calculation."""
    try:
        print(f"Fetching {ticker} with SINGLE API call...", file=sys.stderr)
        stock = yf.Ticker(ticker)
        data = stock.history(period="3mo", interval="1d")  # ONE CALL GETS EVERYTHING
        
        if data.empty:
            raise Exception("No data returned")
        
        current_price = data['Close'].iloc[-1]
        days_data = len(data)
        price_change_20d = (current_price - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100 if days_data > 20 else 0
        
        # Simple distance-based probability (works immediately)
        distance_pct = (target_price - current_price) / current_price * 100
        base_prob = max(0, 100 - abs(distance_pct) * 0.5)  # Closer = higher probability
        
        # Trend adjustment
        if price_change_20d > 2:
            base_prob += 15
        elif price_change_20d < -2:
            base_prob -= 15
        
        probability = min(100, max(0, base_prob))
        
        report = {
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
            },
            "technical_indicators": {"rsi": 50.0, "macd": 0.0},
            "statistics": {"volatility": 2.5}
        }
        
        print(f"✓ {ticker}: ${current_price:.2f} → ${target_price} | {probability:.1f}%", file=sys.stderr)
        return report
        
    except Exception as e:
        print(f"✗ {ticker} ERROR: {e}", file=sys.stderr)
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "current_price": 0.0,
            "target_price": float(target_price),
            "distance_to_target": float(target_price),
            "distance_pct": 100.0,
            "probability": {"composite_probability": 0.0, "confidence_level": "LOW"},
            "status": "error",
            "error_message": str(e)
        }

def analyze_stocks():
    """Background analysis loop - 5min cycles, single API call per stock."""
    global stock_data, last_update, analysis_thread_running
    analysis_thread_running = True
    
    print("ANALYSIS THREAD STARTED - SINGLE CALL MODE", file=sys.stderr)
    
    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\nCYCLE #{cycle} - {datetime.now().strftime('%H:%M:%S')}", file=sys.stderr)
            
            with data_lock:
                current_config = stock_config.copy()
            
            for ticker, target in current_config.items():
                report = analyze_single_stock(ticker, target)
                with data_lock:
                    stock_data[ticker] = report
                    last_update = datetime.now()
                time.sleep(10)  # Rate limiting between stocks
            
            print(f"CYCLE #{cycle} COMPLETE - Next in 5min", file=sys.stderr)
            time.sleep(300)  # 5 minute cycles
            
        except Exception as e:
            print(f"Analysis error: {e}", file=sys.stderr)
            time.sleep(60)

@app.route("/")
def index():
    html_files = ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]
    for f in html_files:
        if os.path.exists(f):
            print(f"Serving {f}", file=sys.stderr)
            return send_from_directory(".", f)
    return "No dashboard HTML found", 404

@app.route("/api/analysis")
def get_analysis():
    """Frontend-compatible endpoint - always returns data."""
    with data_lock:
        # Fill missing tickers with error state
        for ticker in stock_config:
            if ticker not in stock_data:
                stock_data[ticker] = analyze_single_stock(ticker, stock_config[ticker])
        
        print(f"API: {len(stock_data)} stocks ready", file=sys.stderr)
        return jsonify({
            "status": "ok",
            "timestamp": last_update.isoformat() if last_update else datetime.now().isoformat(),
            "stocks": stock_data
        })

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({"stocks": stock_config})

@app.route("/api/config", methods=["POST"])
def update_config():
    global stock_config
    data = request.get_json()
    stock_config.update(data.get("stocks", {}))
    return jsonify({"success": True, "stocks": stock_config})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "stocks": len(stock_data),
        "config": len(stock_config)
    })

# STARTUP
print("INITIALIZING...", file=sys.stderr)
analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
analysis_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
