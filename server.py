"""
Flask Server for Stock Tracker - EMERGENCY VERSION
Returns mock data if analyzer fails, so you can see if the connection works
"""
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time

# Try to import stock analyzer
try:
    from stock_analyzer import StockAnalyzer
    ANALYZER_AVAILABLE = True
    print("✓ StockAnalyzer imported successfully")
except Exception as e:
    print(f"✗ Error importing StockAnalyzer: {e}")
    print("Will use mock data instead")
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

def create_mock_data(ticker, target_price):
    """Create mock data for testing when analyzer fails"""
    import random
    current_price = random.uniform(100, 300)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'target_price': target_price,
        'distance_to_target': round(target_price - current_price, 2),
        'distance_pct': round(((target_price - current_price) / current_price) * 100, 2),
        'probability': {
            'composite_probability': random.uniform(30, 70),
            'momentum_score': random.uniform(40, 60),
            'statistical_probability': random.uniform(35, 65),
            'ml_probability': random.uniform(40, 60),
            'distance_factor': random.uniform(30, 70),
            'time_factor': 50.0,
            'confidence_level': 'MEDIUM',
            'component_weights': {
                'momentum': 0.25,
                'monte_carlo': 0.30,
                'ml': 0.30,
                'distance': 0.10,
                'time': 0.05
            }
        },
        'technical_indicators': {
            'rsi': round(random.uniform(30, 70), 2),
            'macd': round(random.uniform(-2, 2), 4),
            'macd_signal': round(random.uniform(-2, 2), 4),
            'stoch_k': round(random.uniform(20, 80), 2),
            'adx': round(random.uniform(15, 35), 2),
            'cci': round(random.uniform(-100, 100), 2),
            'mfi': round(random.uniform(30, 70), 2),
            'williams_r': round(random.uniform(-80, -20), 2),
            'volume_ratio': round(random.uniform(0.8, 1.5), 2),
            'trend_signals': {
                'price_above_sma20': random.choice([True, False]),
                'price_above_sma50': random.choice([True, False]),
                'price_above_sma200': random.choice([True, False]),
                'golden_cross': random.choice([True, False]),
                'macd_bullish': random.choice([True, False]),
                'rsi_neutral': True,
                'strong_trend': random.choice([True, False])
            }
        },
        'statistics': {
            'annual_volatility': round(random.uniform(25, 50), 2),
            'sharpe_ratio': round(random.uniform(0.5, 2.0), 2),
            'sortino_ratio': round(random.uniform(0.5, 2.0), 2),
            'max_drawdown': round(random.uniform(-30, -10), 2),
            'return_1d': round(random.uniform(-3, 3), 2),
            'return_5d': round(random.uniform(-5, 5), 2),
            'return_20d': round(random.uniform(-10, 10), 2),
            'expected_price_median': round(current_price * random.uniform(0.9, 1.1), 2),
            'price_range_90': [
                round(current_price * 0.8, 2),
                round(current_price * 1.2, 2)
            ]
        },
        'machine_learning': {
            'probability': round(random.uniform(35, 65), 2),
            'predicted_return': round(random.uniform(-5, 10), 2),
            'confidence': 'medium'
        }
    }

def analyze_stocks():
    """Background task to analyze stocks"""
    global stock_data, last_update
    
    print("\n" + "="*60)
    print("STARTING ANALYSIS THREAD")
    print("="*60)
    
    if not ANALYZER_AVAILABLE:
        print("⚠️  Analyzer not available - using MOCK DATA for testing")
        print("This allows you to test the frontend while we fix the analyzer")
        
        # Generate mock data once
        with data_lock:
            for ticker, target_price in stock_config.items():
                stock_data[ticker] = create_mock_data(ticker, target_price)
            last_update = datetime.now()
        
        print(f"✓ Generated mock data for {len(stock_data)} stocks")
        
        # Keep updating with slight variations
        while True:
            time.sleep(10)
            with data_lock:
                for ticker in stock_config.keys():
                    stock_data[ticker] = create_mock_data(ticker, stock_config[ticker])
                last_update = datetime.now()
            print(f"Updated mock data at {last_update.strftime('%H:%M:%S')}")
        
        return
    
    # Real analysis (if analyzer is available)
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
                        print(f"✗ {ticker}: Failed - using mock data")
                        new_data[ticker] = create_mock_data(ticker, target_price)
                    
                    time.sleep(3)
                    
                except Exception as e:
                    print(f"✗ Error {ticker}: {e} - using mock data")
                    new_data[ticker] = create_mock_data(ticker, target_price)
            
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
            print(f"Serving {filename}")
            return send_from_directory('.', filename)
    
    print(f"ERROR: No HTML file found. Checked: {html_files}")
    return f"Error: No HTML file found. Looking for: {html_files}", 404

@app.route('/api/analysis')
def get_analysis():
    """Return stock data"""
    print(f"API Request: /api/analysis - stocks in cache: {list(stock_data.keys())}")
    
    with data_lock:
        response_data = {
            'timestamp': last_update.isoformat() if last_update else datetime.now().isoformat(),
            'stocks': stock_data
        }
        
        print(f"Returning data for {len(stock_data)} stocks")
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
        health_data = {
            'status': 'healthy',
            'stocks_configured': len(stock_config),
            'stocks_analyzed': len(stock_data),
            'last_update': last_update.isoformat() if last_update else None,
            'analyzer_available': ANALYZER_AVAILABLE,
            'using_mock_data': not ANALYZER_AVAILABLE
        }
        print(f"Health check: {health_data}")
        return jsonify(health_data)

@app.route('/api/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat(),
        'server_running': True
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("STOCK TRACKER SERVER - EMERGENCY MODE")
    print("="*60)
    print(f"Analyzer Available: {ANALYZER_AVAILABLE}")
    
    if not ANALYZER_AVAILABLE:
        print("⚠️  WARNING: Running with MOCK DATA")
        print("This is for testing the frontend connection")
    
    print(f"Configured Stocks: {list(stock_config.keys())}")
    
    # Start analysis thread
    analysis_thread = threading.Thread(target=analyze_stocks, daemon=True)
    analysis_thread.start()
    print("✓ Analysis thread started")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Port: {port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
