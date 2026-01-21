"""
Flask Server for Stock Tracker
Simple version - serves HTML and provides analysis API
"""
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
import traceback

# Import stock analyzer
try:
    from stock_analyzer import StockAnalyzer
    ANALYZER_AVAILABLE = True
    print("✓ StockAnalyzer imported successfully")
except Exception as e:
    print(f"✗ Error importing StockAnalyzer: {e}")
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

def analyze_stocks():
    """Background task to analyze stocks"""
    global stock_data, last_update
    
    if not ANALYZER_AVAILABLE:
        print("Analyzer not available")
        return
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"Analysis cycle: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            with data_lock:
                current_config = stock_config.copy()
            
            new_data = {}
            
            for ticker, target_price in current_config.items():
                try:
                    print(f"\nAnalyzing {ticker} (Target: ${target_price})...")
                    analyzer = StockAnalyzer(ticker=ticker, target_price=target_price)
                    report = analyzer.generate_analysis_report()
                    
                    if report:
                        new_data[ticker] = report
                        prob = report['probability']['composite_probability']
                        print(f"✓ {ticker}: {prob:.1f}%")
                    else:
                        print(f"✗ {ticker}: Failed")
                        with data_lock:
                            if ticker in stock_data:
                                new_data[ticker] = stock_data[ticker]
                    
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"✗ Error {ticker}: {e}")
                    with data_lock:
                        if ticker in stock_data:
                            new_data[ticker] = stock_data[ticker]
            
            with data_lock:
                stock_data = new_data
                last_update = datetime.now()
            
            print(f"\nComplete: {len(new_data)}/{len(current_config)} stocks")
            time.sleep(10)
            
        except Exception as e:
            print(f"Error in analysis loop: {e}")
            time.sleep(10)

@app.route('/')
def index():
    """Serve HTML file"""
    html_files = ['dashboard_multi.html', 'dashboard_multi_with_settings.html', 'index.html']
    
    for filename in html_files:
        if os.path.exists(filename):
            return send_from_directory('.', filename)
    
    return "No HTML file found", 404

@app.route('/api/analysis')
def get_analysis():
    """Return stock data"""
    with data_lock:
        return jsonify({
            'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
            'stocks': stock_data
        })

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
        
        print(f"\nConfig updated: {list(new_config.keys())}")
        
        return jsonify({
            'success': True,
            'stocks': stock_config
        })
        
    except Exception as e:
        print(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check"""
    with data_lock:
        return jsonify({
            'status': 'healthy',
            'stocks_configured': len(stock_config),
            'stocks_analyzed': len(stock_data),
            'last_update': last_update.isoformat() if last_update else None,
            'analyzer_available': ANALYZER_AVAILABLE
        })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STOCK TRACKER SERVER")
    print("="*60)
    print(f"Analyzer: {ANALYZER_AVAILABLE}")
    print(f"Stocks: {list(stock_config.keys())}")
    
    if ANALYZER_AVAILABLE:
        analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
        analysis_thread.start()
        print("Analysis thread started")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Port: {port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
