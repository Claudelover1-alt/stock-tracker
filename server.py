"""
Flask Server for Stock Tracker
Handles API requests and serves real-time stock analysis
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
from stock_analyzer import StockAnalyzer

app = Flask(__name__, static_folder='.')
CORS(app)

# Global storage for stock data and configuration
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
    """Background task to analyze all configured stocks"""
    global stock_data, last_update
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"Starting analysis cycle at {datetime.now().strftime('%H:%M:%S')}")
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
                        print(f"✓ {ticker}: {prob:.1f}% probability - Success!")
                    else:
                        print(f"✗ {ticker}: Analysis failed")
                        # Keep old data if available
                        with data_lock:
                            if ticker in stock_data:
                                new_data[ticker] = stock_data[ticker]
                    
                    # Small delay between stocks to avoid rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"✗ Error analyzing {ticker}: {e}")
                    # Keep old data if available
                    with data_lock:
                        if ticker in stock_data:
                            new_data[ticker] = stock_data[ticker]
            
            # Update global data
            with data_lock:
                stock_data = new_data
                last_update = datetime.now()
            
            print(f"\n{'='*60}")
            print(f"Analysis cycle complete: {len(new_data)}/{len(current_config)} stocks updated")
            print(f"{'='*60}\n")
            
            # Wait 10 seconds before next update cycle
            time.sleep(10)
            
        except Exception as e:
            print(f"Error in analysis loop: {e}")
            time.sleep(10)

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'dashboard_multi.html')

@app.route('/api/analysis')
def get_analysis():
    """Return current stock analysis data"""
    with data_lock:
        if not stock_data:
            return jsonify({
                'error': 'No data available yet',
                'timestamp': datetime.now().isoformat(),
                'stocks': {}
            }), 503
        
        return jsonify({
            'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
            'stocks': stock_data
        })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Return current stock configuration"""
    with data_lock:
        return jsonify({
            'stocks': stock_config
        })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update stock configuration"""
    global stock_config
    
    try:
        data = request.get_json()
        new_config = data.get('stocks', {})
        
        # Validate the configuration
        if not isinstance(new_config, dict):
            return jsonify({'error': 'Invalid configuration format'}), 400
        
        for ticker, target in new_config.items():
            if not isinstance(ticker, str) or not isinstance(target, (int, float)):
                return jsonify({'error': f'Invalid data for {ticker}'}), 400
            if target <= 0:
                return jsonify({'error': f'Target price must be positive for {ticker}'}), 400
        
        # Update configuration
        with data_lock:
            stock_config = new_config
            # Clear old data for removed stocks
            stocks_to_remove = [t for t in stock_data.keys() if t not in stock_config]
            for ticker in stocks_to_remove:
                del stock_data[ticker]
        
        print(f"\n{'='*60}")
        print(f"Configuration updated:")
        print(f"New stocks: {list(new_config.keys())}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'stocks': stock_config
        })
        
    except Exception as e:
        print(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    with data_lock:
        return jsonify({
            'status': 'healthy',
            'stocks_configured': len(stock_config),
            'stocks_analyzed': len(stock_data),
            'last_update': last_update.isoformat() if last_update else None
        })

if __name__ == '__main__':
    # Start background analysis thread
    analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
    analysis_thread.start()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
