"""
Flask Server for Stock Tracker
Handles API requests and serves real-time stock analysis
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

# Create Flask app
app = Flask(__name__, static_folder=".")
CORS(app)

# Temporary debug route to confirm this server.py is live
@app.route("/ping-debug-123")
def ping_debug_123():
    return "PING from *this* server.py", 200

# Global storage for stock data and configuration
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


def analyze_stocks():
    """Background task to analyze all configured stocks"""
    global stock_data, last_update, analysis_running

    if not ANALYZER_AVAILABLE:
        print("Stock analyzer not available - skipping analysis")
        return

    analysis_running = True

    while True:
        try:
            print("\n" + "=" * 60)
            print(f"Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            # Snapshot config
            with data_lock:
                current_config = stock_config.copy()

            print(f"Configured stocks: {list(current_config.keys())}")

            new_data = {}

            for ticker, target_price in current_config.items():
                try:
                    print(f"\nAnalyzing {ticker} (Target: ${target_price})...")
                    analyzer = StockAnalyzer(ticker=ticker, target_price=target_price)
                    report = analyzer.generate_analysis_report()

                    if report:
                        new_data[ticker] = report
                        # composite_probability is inside report['probability']
                        prob_block = report.get("probability", {})
                        prob = prob_block.get("composite_probability", 0.0)
                        print(f"✓ {ticker}: {prob:.1f}% probability - Success!")
                    else:
                        print(f"✗ {ticker}: Analysis failed - no report generated")
                        # Keep old data if available
                        with data_lock:
                            if ticker in stock_data:
                                new_data[ticker] = stock_data[ticker]
                                print(f"  Using cached data for {ticker}")

                    # Small delay between stocks to avoid rate limiting
                    time.sleep(3)

                except Exception as e:
                    print(f"✗ Error analyzing {ticker}: {e}")
                    print(traceback.format_exc())
                    # Keep old data if available
                    with data_lock:
                        if ticker in stock_data:
                            new_data[ticker] = stock_data[ticker]
                            print(f"  Using cached data for {ticker}")

            # Update global data
            with data_lock:
                stock_data.clear()
                stock_data.update(new_data)
                last_update = datetime.now()

            print("\n" + "=" * 60)
            print(f"Analysis cycle complete: {len(new_data)}/{len(current_config)} stocks updated")
            print(f"Last update: {last_update.strftime('%H:%M:%S')}")
            print("=" * 60 + "\n")

            # Wait 10 seconds before next update cycle
            time.sleep(10)

        except Exception as e:
            print(f"Error in analysis loop: {e}")
            print(traceback.format_exc())
            time.sleep(10)


# Start background analysis thread when the module is imported (works under Gunicorn)
if ANALYZER_AVAILABLE:
    try:
        print("Starting analysis thread at import time")
        analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
        analysis_thread.start()
    except Exception as e:
        print(f"Error starting analysis thread: {e}")
else:
    print("WARNING: Analysis thread not started - StockAnalyzer not available")


@app.route("/")
def index():
    """Serve the main HTML file"""
    try:
        html_files = ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]
        for filename in html_files:
            if os.path.exists(filename):
                print(f"Serving {filename}")
                return send_from_directory(".", filename)

        return f"Error: No HTML file found. Looking for: {html_files}", 404
    except Exception as e:
        print(f"Error serving index: {e}")
        return f"Error: {e}", 500


@app.route("/api/analysis")
def get_analysis():
    """Return current stock analysis data - FRONTEND COMPATIBLE FORMAT"""
    try:
        with data_lock:
            print(f"API request - stocks in cache: {list(stock_data.keys())}")
            
            # Frontend expects FLAT array of stocks, not nested object
            if stock_data:
                # Flatten: convert stocks[ticker] → stocks[{ticker, probability, ...}]
                all_stocks = []
                for ticker, report in stock_data.items():
                    flat_report = {
                        'ticker': ticker,
                        'current_price': report.get('current_price', 0),
                        'target_price': report.get('target_price', 0),
                        'distance_to_target': report.get('distance_to_target', 0),
                        'distance_pct': report.get('distance_pct', 0),
                        'probability': report.get('probability', {}).get('composite_probability', 0),
                        'rsi': report.get('technical_indicators', {}).get('rsi', 50),
                        'confidence': report.get('probability', {}).get('confidence_level', 'LOW'),
                        'status': 'ready'
                    }
                    all_stocks.append(flat_report)
                
                return jsonify({
                    'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
                    'stocks': all_stocks,  # Array format frontend expects
                    'status': 'ready',
                    'analyzer_available': ANALYZER_AVAILABLE,
                    'analysis_running': analysis_running,
                })
            else:
                # During startup, return loading state with empty array (not empty object)
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'stocks': [],  # Empty ARRAY, not empty object
                    'status': 'loading',  # Frontend-friendly status
                    'analyzer_available': ANALYZER_AVAILABLE,
                    'analysis_running': analysis_running,
                })
    except Exception as e:
        print(f"Error in get_analysis: {e}")
        print(traceback.format_ex
