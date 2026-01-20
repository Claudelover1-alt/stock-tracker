"""
Flask Server for Stock Tracker - TIMEOUT-SAFE VERSION
Fast initial load + background analysis
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import threading
import time
import traceback

# Try to import stock analyzer
try:
    from stock_analyzer import StockAnalyzer
    ANALYZER_AVAILABLE = True
    print("✓ StockAnalyzer imported successfully")
except Exception as e:
    print(f"✗ Error importing StockAnalyzer: {e}")
    ANALYZER_AVAILABLE = False

app = Flask(__name__, static_folder=".")
CORS(app)

stock_data = {}
stock_config = {"TSLA": 500.0, "AAPL": 250.0, "NVDA": 200.0, "RCAT": 30.0}
data_lock = threading.Lock()
last_update = None
analysis_running = False

def analyze_stocks():
    global stock_data, last_update, analysis_running
    if not ANALYZER_AVAILABLE:
        return
        
    analysis_running = True
    
    while True:
        try:
            print(f"Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
            
            with data_lock:
                current_config = stock_config.copy()
            
            new_data = {}
            
            # Analyze only 1-2 stocks per cycle to stay fast
            for ticker in list(current_config.keys())[:2]:
                try:
                    print(f"Analyzing {ticker}...")
                    analyzer = StockAnalyzer(ticker=ticker, target_price=stock_config[ticker])
                    report = analyzer.generate_analysis_report()
                    
                    if report:
                        new_data[ticker] = report
                        print(f"✓ {ticker} analyzed")
                    
                    time.sleep(2)  # Faster cycle
                    
                except Exception as e:
                    print(f"✗ {ticker} failed: {e}")
            
            with data_lock:
                stock_data.update(new_data)
                last_update = datetime.now()
            
            print(f"Cycle complete: {len(new_data)} stocks")
            time.sleep(15)  # Faster refresh
            
        except Exception as e:
            print(f"Analysis error: {e}")
            time.sleep(15)

# Start thread AFTER all routes defined (Gunicorn safe)
if ANALYZER_AVAILABLE:
    analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
    analysis_thread.start()
    print("Analysis thread started")

@app.route("/")
def index():
    html_files = ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]
    for filename in html_files:
        if os.path.exists(filename):
            print(f"Serving {filename}")
            return send_from_directory(".", filename)
    return "Error: No HTML file found", 404

@app.route("/api/analysis")
def get_analysis():
    with data_lock:
        print(f"API - stocks: {list(stock_data.keys())}")
        
        # Fast response - frontend expects array format
        if stock_data:
            stocks_list = []
            for ticker, report in stock_data.items():
                stocks_list.append({
                    'ticker': ticker,
                    'current_price': report.get('current_price', 0),
                    'target_price': stock_config.get(ticker, 0),
                    'probability': report.get('probability', {}).get('composite_probability', 0),
                })
            return jsonify({
                'stocks': stocks_list,
                'status': 'ready',
                'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat()
            })
        else:
            return jsonify({
                'stocks': [],
                'status': 'loading',
                'timestamp': datetime.now().isoformat()
            })

@app.route("/api/health")
def health():
    with data_lock:
        return jsonify({
            "status": "healthy",
            "stocks_configured": len(stock_config),
            "stocks_analyzed": len(stock_data),
            "analyzer_available": ANALYZER_AVAILABLE,
            "analysis_running": analysis_running,
        })

@app.route("/api/test")
def test():
    return jsonify({"message": "API working!", "timestamp": datetime.now().isoformat()})

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({"stocks": stock_config})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
