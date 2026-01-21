# server.py
# Flask Server for Stock Tracker - ROBUST VERSION
# - Never returns empty stocks list while analysis is running
# - Caches a minimal "error" report when full analysis fails

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
import sys

print("=" * 60, file=sys.stderr)
print("IMPORTING STOCK ANALYZER...", file=sys.stderr)
print("=" * 60, file=sys.stderr)

try:
    # IMPORTANT: adjust name if your file is stockanalyzer.py (no underscore)
    from stockanalyzer import StockAnalyzer
    ANALYZER_AVAILABLE = True
    print("StockAnalyzer imported successfully", file=sys.stderr)
except Exception as e:
    print(f"Error importing StockAnalyzer: {e}", file=sys.stderr)
    ANALYZER_AVAILABLE = False

app = Flask(__name__, static_folder=".")
CORS(app)

# -------------------------------------------------------------------
# GLOBAL STATE
# -------------------------------------------------------------------

# Default configuration
stock_config = {
    "TSLA": 500.0,
    "AAPL": 250.0,
    "NVDA": 200.0,
    "RCAT": 30.0,
}

# Latest report per ticker
stock_data = {}          # ticker -> report dict
data_lock = threading.Lock()
last_update = None
analysis_thread_running = False


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

def make_error_report(ticker: str, target_price: float, reason: str) -> dict:
    """Create a minimal report the frontend can still render."""
    now = datetime.now().isoformat()
    return {
        "timestamp": now,
        "ticker": ticker,
        "current_price": None,
        "target_price": float(target_price),
        "distance_to_target": None,
        "distance_pct": None,
        "probability": {
            "composite_probability": 0.0,
            "momentum_score": 0.0,
            "statistical_probability": 0.0,
            "ml_probability": 0.0,
            "distance_factor": 0.0,
            "time_factor": 0.0,
            "confidence_level": "LOW",
        },
        "technical_indicators": {},
        "statistics": {},
        "machine_learning": {
            "probability": 0.0,
            "predicted_return": 0.0,
            "confidence": "low",
        },
        "status": "error",
        "error_message": reason,
    }


def wait_for_initial_data(timeout_seconds: int = 45) -> bool:
    """
    Block for up to timeout_seconds until either:
    - at least one stock has a report in stock_data, OR
    - the analysis thread has clearly produced error reports.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with data_lock:
            if stock_data:
                return True
        time.sleep(0.5)
    return False


# -------------------------------------------------------------------
# BACKGROUND ANALYSIS THREAD
# -------------------------------------------------------------------

def analyze_stocks():
    """Analyze all configured stocks in a loop and cache their reports."""
    global stock_data, last_update, analysis_thread_running
    analysis_thread_running = True

    print("=" * 60, file=sys.stderr)
    print("ANALYSIS THREAD IS NOW RUNNING!", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if not ANALYZER_AVAILABLE:
        print("Analyzer not available, exiting analysis thread", file=sys.stderr)
        analysis_thread_running = False
        return

    cycle_count = 0

    while True:
        try:
            cycle_count += 1
            print("=" * 60, file=sys.stderr)
            print(f"ANALYSIS CYCLE #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            with data_lock:
                current_config = stock_config.copy()
                print(f"Stocks to analyze: {list(current_config.keys())}", file=sys.stderr)

            stocks_completed = 0

            for ticker, target_price in current_config.items():
                try:
                    print(f"--- Analyzing {ticker} (Target: ${target_price}) ---", file=sys.stderr)
                    analyzer = StockAnalyzer(ticker=ticker, target_price=target_price)

                    # If generate_analysis_report raises or returns None,
                    # we will catch and still store an error report below.
                    report = analyzer.generate_analysis_report()

                    if report:
                        with data_lock:
                            stock_data[ticker] = report
                            last_update = datetime.now()
                            prob = report.get("probability", {}).get("composite_probability", 0.0)
                            print(f"✓ {ticker} COMPLETE: {prob:.1f}% probability", file=sys.stderr)
                            print(f"💾 SAVED {ticker} to cache immediately", file=sys.stderr)
                            print(f"📊 Cache now contains: {list(stock_data.keys())}", file=sys.stderr)
                            stocks_completed += 1
                    else:
                        # Generate a minimal error report so frontend still sees the ticker
                        msg = "No report generated (None returned)"
                        print(f"✗ {ticker} FAILED: {msg}", file=sys.stderr)
                        with data_lock:
                            stock_data[ticker] = make_error_report(ticker, target_price, msg)
                            last_update = datetime.now()
                            print(f"💾 SAVED ERROR REPORT for {ticker} to cache", file=sys.stderr)
                            print(f"📊 Cache now contains: {list(stock_data.keys())}", file=sys.stderr)

                    time.sleep(3)

                except Exception as e:
                    # Hard failure inside analysis for this ticker
                    msg = f"Exception in analysis: {e}"
                    print(f"✗ {ticker} FAILED: {msg}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)

                    with data_lock:
                        stock_data[ticker] = make_error_report(ticker, target_price, msg)
                        last_update = datetime.now()
                        print(f"💾 SAVED ERROR REPORT for {ticker} to cache", file=sys.stderr)
                        print(f"📊 Cache now contains: {list(stock_data.keys())}", file=sys.stderr)

            print("=" * 60, file=sys.stderr)
            print(f"CYCLE #{cycle_count} COMPLETE", file=sys.stderr)
            print(f"Successfully analyzed and saved: {stocks_completed}/{len(current_config)} stocks", file=sys.stderr)
            with data_lock:
                print(f"Final cache contains: {list(stock_data.keys())}", file=sys.stderr)
                if last_update:
                    print(f"Last update: {last_update.strftime('%H:%M:%S')}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            print("Waiting 15 seconds before next cycle...", file=sys.stderr)
            time.sleep(15)

        except Exception as e:
            print(f"FATAL ERROR in analysis loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            time.sleep(15)


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/")
def index():
    """Serve main dashboard HTML."""
    html_files = [
        "dashboard_multi.html",
        "dashboard_multi_with_settings.html",
        "index.html",
    ]
    for filename in html_files:
        if os.path.exists(filename):
            print(f"Serving {filename}", file=sys.stderr)
            return send_from_directory(".", filename)

    print(f"ERROR: No HTML file found. Checked {html_files}", file=sys.stderr)
    return f"Error: No HTML file found. Looking for {html_files}", 404


@app.route("/api/analysis")
def get_analysis():
    """
    Return cached analysis.
    Guarantees:
      - If analyzer is up and running, never returns an empty 'stocks' dict
        for configured tickers; at minimum you get error reports.
    """
    if not ANALYZER_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "Analyzer not available on server",
            "timestamp": datetime.now().isoformat(),
            "stocks": {},
        }), 500

    # Wait for initial population (either normal or error reports)
    data_ready = wait_for_initial_data(timeout_seconds=45)

    with data_lock:
        # Ensure every configured ticker has an entry in stock_data
        for ticker, target_price in stock_config.items():
            if ticker not in stock_data and data_ready:
                # If the thread somehow skipped it, create a placeholder
                stock_data[ticker] = make_error_report(
                    ticker, target_price, "No data yet from analysis thread"
                )

        num_stocks = len(stock_data)
        stocks_list = list(stock_data.keys())
        print(f"API Request: /api/analysis - stocks in cache: {stocks_list}", file=sys.stderr)

        if not data_ready and num_stocks == 0:
            # Analysis not yet produced any data
            response_data = {
                "status": "pending",
                "message": "Analysis running, no data yet",
                "timestamp": datetime.now().isoformat(),
                "stocks": {},
            }
            print("Returning PENDING (no data yet) from /api/analysis", file=sys.stderr)
            return jsonify(response_data)

        response_data = {
            "status": "ok",
            "timestamp": last_update.isoformat() if last_update else datetime.now().isoformat(),
            "stocks": stock_data,
        }
        print(f"Returning data for {num_stocks} stocks: {stocks_list}", file=sys.stderr)
        return jsonify(response_data)


@app.route("/api/config", methods=["GET"])
def get_config():
    with data_lock:
        return jsonify({"stocks": stock_config})


@app.route("/api/config", methods=["POST"])
def update_config():
    global stock_config
    try:
        data = request.get_json()
        new_config = data.get("stocks", {})
        print(f"Received config update: {new_config}", file=sys.stderr)

        if not isinstance(new_config, dict):
            return jsonify({"error": "Invalid format"}), 400

        for ticker, target in new_config.items():
            if not isinstance(ticker, str) or not isinstance(target, (int, float)):
                return jsonify({"error": f"Invalid data for {ticker}"}), 400
            if target <= 0:
                return jsonify({"error": f"Invalid target for {ticker}"}), 400

        with data_lock:
            stock_config = new_config
            # Drop cached data for removed tickers
            to_remove = [t for t in list(stock_data.keys()) if t not in stock_config]
            for t in to_remove:
                del stock_data[t]
            print(f"Config updated: {list(new_config.keys())}", file=sys.stderr)

        return jsonify({"success": True, "stocks": stock_config})

    except Exception as e:
        print(f"Error updating config: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    with data_lock:
        data = {
            "status": "healthy" if ANALYZER_AVAILABLE else "degraded",
            "stocks_configured": len(stock_config),
            "stocks_analyzed": len(stock_data),
            "last_update": last_update.isoformat() if last_update else None,
            "analyzer_available": ANALYZER_AVAILABLE,
            "analysis_thread_running": analysis_thread_running,
        }
        print(f"Health check: {data}", file=sys.stderr)
        return jsonify(data)


@app.route("/api/test")
def test():
    return jsonify({
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "server_running": True,
    })


# -------------------------------------------------------------------
# STARTUP
# -------------------------------------------------------------------

print("=" * 60, file=sys.stderr)
print("STOCK TRACKER SERVER INITIALIZATION", file=sys.stderr)
print("=" * 60, file=sys.stderr)
print(f"Analyzer Available: {ANALYZER_AVAILABLE}", file=sys.stderr)
print(f"Configured Stocks: {list(stock_config.keys())}", file=sys.stderr)

if ANALYZER_AVAILABLE:
    print("STARTING ANALYSIS THREAD...", file=sys.stderr)
    analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
    analysis_thread.start()
    print("ANALYSIS THREAD STARTED!", file=sys.stderr)
    time.sleep(1)
    if analysis_thread.is_alive():
        print("Thread confirmed running", file=sys.stderr)
    else:
        print("WARNING: Thread may have stopped", file=sys.stderr)
else:
    print("Analyzer not available - no analysis thread", file=sys.stderr)

print("=" * 60, file=sys.stderr)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
else:
    print("RUNNING UNDER GUNICORN", file=sys.stderr)
    print("Workers should be set to 1", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
