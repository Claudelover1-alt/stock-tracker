"""
Flask Server for Stock Tracker - DEBUG VERSION
Shows exactly why analysis thread isn't starting
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import threading
import time
import traceback

print("=== SERVER STARTING ===")

# Try to import stock analyzer
try:
    print("Attempting to import StockAnalyzer...")
    from stock_analyzer import StockAnalyzer
    print("✓ StockAnalyzer imported successfully!")
    ANALYZER_AVAILABLE = True
except Exception as e:
    print(f"✗ FAILED to import StockAnalyzer: {e}")
    print(traceback.format_exc())
    ANALYZER_AVAILABLE = False

print(f"ANALYZER_AVAILABLE = {ANALYZER_AVAILABLE}")

# Create Flask app
app = Flask(__name__, static_folder=".")
CORS(app)

print("Flask app created")

# Debug route
@app.route("/ping-debug-123")
def ping_debug_123():
    return "PING from server.py", 200

# Global storage
stock_data = {}
stock_config = {
    "TSLA": 500.0,
    "AAPL": 250.0,
    "NVDA": 200.0,
    "RCAT": 30.0,
}
data_lock = threading.Lock()
last_update = None
analysis_running = False

print(f"Initial config: {list(stock_config.keys())}")

def analyze_stocks():
    """Background task to analyze all configured stocks"""
    global stock_data, last_update, analysis_running
    
    print("*** THREAD STARTED: analyze_stocks() ***")
    
    if not ANALYZER_AVAILABLE:
        print("*** THREAD EXITING: analyzer not available ***")
        return

    print("*** Setting analysis_running = True ***")
    analysis_running = True

    while True:
        try:
            print("\n" + "=" * 60)
            print(f"Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            with data_lock:
                current_config = stock_config.copy()
            print(f"Configured stocks: {list(current_config.keys())}")

            new_data = {}
            
            # Test just ONE stock first
            for ticker, target_price in list(current_config.items())[:1]:  # Only TSLA
                try:
                    print(f"\n*** Testing {ticker} (Target: ${target_price})...")
                    analyzer = StockAnalyzer(ticker=ticker, target_price=target_price)
                    print(f"*** StockAnalyzer created for {ticker} ***")
                    report = analyzer.generate_analysis_report()
                    print(f"*** generate_analysis_report() returned: {type(report)} ***")

                    if report:
                        new_data[ticker] = report
                        prob_block = report.get("probability", {})
                        prob = prob_block.get("composite_probability", 0.0)
                        print(f"✓ {ticker}: {prob:.1f}% probability - SUCCESS!")
                    else:
                        print(f"✗ {ticker}: No report returned")

                    time.sleep(3)
                    break  # Only test one stock

                except Exception as e:
                    print(f"✗ Error analyzing {ticker}: {e}")
                    print(traceback.format_exc())

            with data_lock:
                stock_data.clear()
                stock_data.update(new_data)
                last_update = datetime.now()

            print(f"*** Cycle complete: {len(new_data)} stocks updated ***")
            print("=" * 60 + "\n")
            time.sleep(10)

        except Exception as e:
            print(f"Error in analysis loop: {e}")
            print(traceback.format_exc())
            time.sleep(10)


print("*** About to start analysis thread ***")
# Start background analysis thread
if ANALYZER_AVAILABLE:
    try:
        print("*** Creating thread object ***")
        analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
        print("*** Starting thread ***")
        analysis_thread.start()
        print("*** THREAD STARTED SUCCESSFULLY ***")
    except Exception as e:
        print(f"*** THREAD FAILED TO START: {e} ***")
        print(traceback.format_exc())
else:
    print("*** Skipping thread - analyzer not available ***")


@app.route("/")
def index():
    html_files = ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]
    for filename in html_files:
        if os.path.exists(filename):
            print(f"Serving {filename}")
            return send_from_directory(".", filename)
    return f"Error: No HTML file found. Looking for: {html_files}", 404


@app.route("/api/analysis")
def get_analysis():
    try:
        with data_lock:
            print(f"API request - stocks in cache: {list(stock_data.keys())}")
            if stock_data:
                all_stocks = []
                for ticker, report in stock_data.items():
                    flat_report = {
                        'ticker': ticker,
                        'current_price': report.get('current_price', 0),
                        'target_price': report.get('target_price', 0),
                        'probability': report.get('probability', {}).get('composite_probability', 0),
                    }
                    all_stocks.append(flat_report)
                return jsonify({
                    'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
                    'stocks': all_stocks,
                    'status': 'ready',
                    'analyzer_available': ANALYZER_AVAILABLE,
                    'analysis_running': analysis_running,
                })
            else:
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'stocks': [],
                    'status': 'loading',
                    'analyzer_available': ANALYZER_AVAILABLE,
                    'analysis_running': analysis_running,
                })
    except Exception as e:
        print(f"Error in get_analysis: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'stocks': []}), 500


@app.route("/api/health")
def health():
    try:
        with data_lock:
            return jsoni
