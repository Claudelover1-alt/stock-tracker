"""
Flask Server for Stock Tracker - GUARANTEED WORKING VERSION
Forces analysis thread to start and provides extensive logging
"""
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
import sys

# Import stock analyzer
print("="*60, file=sys.stderr)
print("IMPORTING STOCK ANALYZER...", file=sys.stderr)
print("="*60, file=sys.stderr)

try:
    from stock_analyzer import StockAnalyzer
    ANALYZER_AVAILABLE = True
    print("✓ StockAnalyzer imported successfully", file=sys.stderr)
except Exception as e:
    print(f"✗ Error importing StockAnalyzer: {e}", file=sys.stderr)
    ANALYZER_AVAILABLE = False

app = Flask(__name__, static_folder='.')
CORS(app)

# Global storage
stock_data = {}
stock_config = {
    'TSLA': 500.0,
    'AAPL': 250.0,
    'NVDA': 200.0,
    'RCAT': 30.0
}
data_lock = threading.Lock()
last_update = None
analysis_thread_running = False

def analyze_stocks():
    """Background task to analyze stocks"""
    global stock_data, last_update, analysis_thread_running
    
    analysis_thread_running = True
    
    print("\n" + "="*60, file=sys.stderr)
    print("ANALYSIS THREAD IS NOW RUNNING!", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    if not ANALYZER_AVAILABLE:
        print("⚠️  Analyzer not available", file=sys.stderr)
        analysis_thread_running = False
        return
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"ANALYSIS CYCLE #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            
            with data_lock:
                current_config = stock_config.copy()
            
            print(f"Stocks to analyze: {list(current_config.keys())}", file=sys.stderr)
            
            new_data = {}
            
            for ticker, target_price in current_config.items():
                try:
                    print(f"\n--- Analyzing {ticker} (Target: ${target_price}) ---", file=sys.stderr)
                    analyzer = StockAnalyzer(ticker=ticker, target_price=target_price)
                    report = analyzer.generate_analysis_report()
                    
                    if report:
                        new_data[ticker] = report
                        prob = report['probability']['composite_probability']
                        print(f"✓ {ticker} COMPLETE: {prob:.1f}% probability", file=sys.stderr)
                    else:
                        print(f"✗ {ticker} FAILED: No report generated", file=sys.stderr)
                        # Keep old data if available
                        with data_lock:
                            if ticker in stock_data:
                                new_data[ticker] = stock_data[ticker]
                                print(f"  → Using cached data for {ticker}", file=sys.stderr)
                    
                    # Delay between stocks
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"✗ {ticker} ERROR: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    
                    # Keep old data if available
                    with data_lock:
                        if ticker in stock_data:
                            new_data[ticker] = stock_data[ticker]
                            print(f"  → Using cached data for {ticker}", file=sys.stderr)
            
            # Update global data
            print(f"\n📊 SAVING DATA TO CACHE...", file=sys.stderr)
            print(f"new_data keys: {list(new_data.keys())}", file=sys.stderr)
            print(f"new_data has {len(new_data)} stocks", file=sys.stderr)
            
            with data_lock:
                stock_data.clear()
                stock_data.update(new_data)
                last_update = datetime.now()
                print(f"stock_data after save: {list(stock_data.keys())}", file=sys.stderr)
            
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"CYCLE #{cycle_count} COMPLETE", file=sys.stderr)
            print(f"Success: {len(new_data)}/{len(current_config)} stocks", file=sys.stderr)
            print(f"Stock data now contains: {len(stock_data)} stocks", file=sys.stderr)
            print(f"Last update: {last_update.strftime('%H:%M:%S')}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            
            # Wait before next cycle
            print(f"Waiting 15 seconds before next cycle...", file=sys.stderr)
            time.sleep(15)
            
        except Exception as e:
            print(f"FATAL ERROR in analysis loop: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            time.sleep(15)

@app.route('/')
def index():
    """Serve HTML file"""
    html_files = ['dashboard_multi.html', 'dashboard_multi_with_settings.html', 'index.html']
    
    for filename in html_files:
        if os.path.exists(filename):
            print(f"Serving {filename}", file=sys.stderr)
            return send_from_directory('.', filename)
    
    print(f"ERROR: No HTML file found. Checked: {html_files}", file=sys.stderr)
    return f"Error: No HTML file found. Looking for: {html_files}", 404

@app.route('/api/analysis')
def get_analysis():
    """Return stock data"""
    print(f"API Request: /api/analysis - stocks in cache: {list(stock_data.keys())}", file=sys.stderr)
    
    with data_lock:
        response_data = {
            'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
            'stocks': stock_data
        }
        
        print(f"Returning data for {len(stock_data)} stocks", file=sys.stderr)
        return jsonify(response_data)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get stock configuration"""
    with data_lock:
        return jsonify({'stocks': stock_config})

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update stock configuration"""
    global stock_config
    
    try:
        data = request.get_json()
        new_config = data.get('stocks', {})
        
        print(f"Received config update: {new_config}", file=sys.stderr)
        
        # Validate
        if not isinstance(new_config, dict):
            return jsonify({'error': 'Invalid format'}), 400
        
        for ticker, target in new_config.items():
            if not isinstance(ticker, str) or not isinstance(target, (int, float)):
                return jsonify({'error': f'Invalid data for {ticker}'}), 400
            if target <= 0:
                return jsonify({'error': f'Invalid target for {ticker}'}), 400
        
        # Update
        with data_lock:
            stock_config = new_config
            stocks_to_remove = [t for t in stock_data.keys() if t not in stock_config]
            for ticker in stocks_to_remove:
                del stock_data[ticker]
        
        print(f"✓ Config updated: {list(new_config.keys())}", file=sys.stderr)
        
        return jsonify({
            'success': True,
            'stocks': stock_config
        })
        
    except Exception as e:
        print(f"Error updating config: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check"""
    with data_lock:
        health_data = {
            'status': 'healthy',
            'stocks_configured': len(stock_config),
            'stocks_analyzed': len(stock_data),
            'last_update': last_update.isoformat() if last_update else None,
            'analyzer_available': ANALYZER_AVAILABLE,
            'analysis_thread_running': analysis_thread_running
        }
        print(f"Health check: {health_data}", file=sys.stderr)
        return jsonify(health_data)

@app.route('/api/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat(),
        'server_running': True
    })

# Start analysis thread BEFORE Flask starts
print("\n" + "="*60, file=sys.stderr)
print("STOCK TRACKER SERVER INITIALIZATION", file=sys.stderr)
print("="*60, file=sys.stderr)
print(f"Analyzer Available: {ANALYZER_AVAILABLE}", file=sys.stderr)
print(f"Configured Stocks: {list(stock_config.keys())}", file=sys.stderr)

if ANALYZER_AVAILABLE:
    print("\n🚀 STARTING ANALYSIS THREAD...", file=sys.stderr)
    analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
    analysis_thread.start()
    print("✓✓✓ ANALYSIS THREAD STARTED ✓✓✓", file=sys.stderr)
    
    # Give it a moment to start
    time.sleep(1)
    
    if analysis_thread.is_alive():
        print("✓ Thread confirmed running", file=sys.stderr)
    else:
        print("✗ WARNING: Thread may have stopped", file=sys.stderr)
else:
    print("✗ Analyzer not available - no analysis thread", file=sys.stderr)

print("="*60, file=sys.stderr)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌐 Starting Flask on port {port}", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    
    # IMPORTANT: Use single worker to avoid shared state issues
    # If using Gunicorn, set workers=1 in your Render config
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
else:
    # Running under Gunicorn
    print("\n⚠️  RUNNING UNDER GUNICORN", file=sys.stderr)
    print("Make sure gunicorn is configured with --workers=1", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
