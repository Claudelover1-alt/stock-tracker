"""
Flask Web Server for Multi-Stock Analysis Dashboard
Provides REST API and serves the mobile-optimized dashboard
Supports tracking up to 6 stocks simultaneously
"""

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from stock_analyzer import StockAnalyzer
import json
import threading
import time
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Global state
current_analyses = {}  # Dict of ticker: analysis_report
analyzers = {}  # Dict of ticker: StockAnalyzer instance
update_lock = threading.Lock()
config = None

# Load configuration and initialize immediately when module is imported
def initialize_on_import():
    """Initialize when the module loads"""
    global config, analyzers, current_analyses
    
    if not load_config():
        print("ERROR: Could not load config. Using default.")
        # Create a default config
        config = {
            "stocks": [
                {"ticker": "RCAT", "target_price": 20.0, "enabled": True},
                {"ticker": "AAPL", "target_price": 250.0, "enabled": True}
            ],
            "update_interval_seconds": 1,
            "max_stocks": 6
        }
    
    # Initialize analyzers
    print("\n" + "="*60)
    print("MULTI-STOCK ANALYSIS DASHBOARD")
    print("="*60)
    initialize_analyzers()
    
    # Initialize first analysis
    initialize_analysis()
    
    # Start background update thread
    update_thread = threading.Thread(target=update_analysis, daemon=True)
    update_thread.start()
    print("Background update thread started!")

# Call initialization immediately
initialize_on_import()

def load_config():
    """Load stock configuration from JSON file"""
    global config
    try:
        with open('stocks_config.json', 'r') as f:
            config = json.load(f)
        return True
    except FileNotFoundError:
        print("ERROR: stocks_config.json not found!")
        print("Please run 'python configure_stocks.py' first to set up your stocks.")
        return False
    except json.JSONDecodeError:
        print("ERROR: stocks_config.json is invalid!")
        return False

def initialize_analyzers():
    """Initialize analyzers for all enabled stocks"""
    global analyzers
    enabled_stocks = [s for s in config['stocks'] if s.get('enabled', True)]
    
    if len(enabled_stocks) > config.get('max_stocks', 6):
        print(f"WARNING: More than {config['max_stocks']} stocks enabled. Using first {config['max_stocks']}.")
        enabled_stocks = enabled_stocks[:config['max_stocks']]
    
    for stock in enabled_stocks:
        ticker = stock['ticker']
        target = stock['target_price']
        analyzers[ticker] = StockAnalyzer(ticker, target)
        print(f"Initialized analyzer for {ticker} (target: ${target})")

def update_analysis():
    """Background thread to update analysis every second"""
    global current_analyses
    
    while True:
        try:
            for ticker, analyzer in analyzers.items():
                try:
                    report = analyzer.generate_analysis_report()
                    
                    with update_lock:
                        current_analyses[ticker] = report
                    
                    prob = report['probability']['composite_probability']
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {ticker}: ${report['current_price']:.2f} → ${report['target_price']:.2f} | Probability: {prob:.1f}%")
                    
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Error updating {ticker}: {e}")
                    # Keep the old data if update fails, don't crash
        
        except Exception as e:
            print(f"Error in update loop: {e}")
        
        # Update based on config
        time.sleep(config.get('update_interval_seconds', 1))

@app.route('/api/analysis')
def get_analysis():
    """Get current analysis data for all stocks"""
    with update_lock:
        if not current_analyses:
            return jsonify({'error': 'Analysis not ready yet'}), 503
        return jsonify({
            'stocks': current_analyses,
            'timestamp': datetime.now().isoformat(),
            'count': len(current_analyses)
        })

@app.route('/api/analysis/<ticker>')
def get_stock_analysis(ticker):
    """Get analysis for a specific stock"""
    ticker = ticker.upper()
    with update_lock:
        if ticker not in current_analyses:
            return jsonify({'error': f'Stock {ticker} not found'}), 404
        return jsonify(current_analyses[ticker])

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    return jsonify(config)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'stocks_tracked': len(analyzers)
    })

@app.route('/')
def dashboard():
    """Serve the mobile-optimized dashboard"""
    html_template = open('dashboard_multi.html', 'r').read()
    return render_template_string(html_template)

def initialize_analysis():
    """Initialize first analysis before starting server"""
    global current_analyses
    print("Initializing analysis for all stocks...")
    
    for ticker, analyzer in analyzers.items():
        try:
            print(f"Analyzing {ticker}...")
            report = analyzer.generate_analysis_report()
            current_analyses[ticker] = report
            print(f"✓ {ticker} analysis complete!")
        except Exception as e:
            print(f"✗ Error analyzing {ticker}: {e}")
            # Create a minimal error report so the dashboard can still load
            current_analyses[ticker] = {
                'ticker': ticker,
                'current_price': 0,
                'target_price': analyzers[ticker].target_price,
                'distance_to_target': 0,
                'distance_pct': 0,
                'probability': {
                    'composite_probability': 0,
                    'confidence_level': 'LOW',
                    'momentum_score': 0,
                    'statistical_probability': 0,
                    'ml_probability': 0
                },
                'technical_indicators': {
                    'rsi': 0,
                    'macd': 0,
                    'macd_signal': 0,
                    'stoch_k': 0,
                    'adx': 0,
                    'cci': 0,
                    'mfi': 0,
                    'williams_r': 0,
                    'volume_ratio': 0,
                    'trend_signals': {
                        'price_above_sma20': False,
                        'price_above_sma50': False,
                        'price_above_sma200': False,
                        'golden_cross': False,
                        'macd_bullish': False,
                        'rsi_neutral': False,
                        'strong_trend': False
                    }
                },
                'statistics': {
                    'annual_volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'max_drawdown': 0,
                    'return_1d': 0,
                    'return_5d': 0,
                    'return_20d': 0,
                    'expected_price_median': 0,
                    'price_range_90': [0, 0]
                },
                'machine_learning': {
                    'probability': 0,
                    'predicted_return': 0,
                    'confidence': 'LOW'
                },
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    if current_analyses:
        print(f"\nInitial analysis complete for {len(current_analyses)} stock(s)!")
    else:
        print("\nWarning: No stocks could be analyzed. Check network connection and stock tickers.")

if __name__ == '__main__':
    # Start Flask server
    print("\n" + "="*60)
    print("Dashboard Server Starting...")
    print("="*60)
    print("Dashboard: http://localhost:5000")
    print("API: http://localhost:5000/api/analysis")
    print(f"\nTracking {len(analyzers)} stock(s) with second-by-second updates")
    print("\nTo access from iPhone:")
    print("1. Find your computer's IP address")
    print("2. Open Safari on iPhone and go to http://[YOUR_IP]:5000")
    print("3. Tap 'Share' button and 'Add to Home Screen'")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
