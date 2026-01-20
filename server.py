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
app = Flask(__name__, static_folder='.')
CORS(app)

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
                        # Use either 'probability' or 'composite_probability' if present
                        prob = report.get("composite_probability") or report.get("probability", 0.0)
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
    """Return current stock analysis data"""
    try:
        with data_lock:
            print(f"API request - stocks in cache: {list(stock_data.keys())}")
            return jsonify(
                {
                    "timestamp": last_update.isoformat() if last_update else datetime.now().isoformat(),
                    "stocks": stock_data,
                    "status": "analyzing" if not stock_data else "ready",
                    "analyzer_available": ANALYZER_AVAILABLE,
                    "analysis_running": analysis_running,
                }
            )
    except Exception as e:
        print(f"Error in get_analysis: {e}")
        print(traceback.format_exc())
        return jsonify(
            {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "stocks": {},
            }
        ), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return current stock configuration"""
    try:
        with data_lock:
            return jsonify({"stocks": stock_config})
    except Exception as e:
        print(f"Error in get_config: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update stock configuration"""
    global stock_config

    try:
        data = request.get_json() or {}
        new_config = data.get("stocks", {})

        print(f"Received config update: {new_config}")

        # Validate the configuration
        if not isinstance(new_config, dict):
            return jsonify({"error": "Invalid configuration format"}), 400

        for ticker, target in new_config.items():
            if not isinstance(ticker, str) or not isinstance(target, (int, float)):
                return jsonify({"error": f"Invalid data for {ticker}"}), 400
            if target <= 0:
                return jsonify({"error": f"Target price must be positive for {ticker}"}), 400

        # Update configuration
        with data_lock:
            stock_config = new_config

            # Clear old data for removed stocks
            stocks_to_remove = [t for t in list(stock_data.keys()) if t not in stock_config]
            for ticker in stocks_to_remove:
                del stock_data[ticker]

        print("\n" + "=" * 60)
        print("Configuration updated:")
        print(f"New stocks: {list(new_config.keys())}")
        print("=" * 60 + "\n")

        return jsonify({"success": True, "stocks": stock_config})
    except Exception as e:
        print(f"Error updating config: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    """Health check endpoint"""
    try:
        with data_lock:
            return jsonify(
                {
                    "status": "healthy",
                    "stocks_configured": len(stock_config),
                    "stocks_analyzed": len(stock_data),
                    "last_update": last_update.isoformat() if last_update else None,
                    "analyzer_available": ANALYZER_AVAILABLE,
                    "analysis_running": analysis_running,
                }
            )
    except Exception as e:
        print(f"Error in health check: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/test")
def test():
    """Simple test endpoint"""
    return jsonify({"message": "API is working!", "timestamp": datetime.now().isoformat()})


# Local development entrypoint only.
# On Render, use gunicorn server:app as the start command.
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STOCK TRACKER SERVER STARTING")
    print("=" * 60)
    print(f"Analyzer available: {ANALYZER_AVAILABLE}")
    print(f"Initial stocks: {list(stock_config.keys())}")

    if ANALYZER_AVAILABLE:
        analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
        analysis_thread.start()
        print("Analysis thread started")
    else:
        print("WARNING: Analysis thread not started - StockAnalyzer not available")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False)
