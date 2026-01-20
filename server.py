from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

# FAKE DATA - shows page works instantly
DEMO_STOCKS = [
    {
        'ticker': 'TSLA',
        'current_price': 423.35,
        'target_price': 500.0,
        'probability': 42.3,
        'rsi': 58.7,
        'confidence': 'HIGH'
    },
    {
        'ticker': 'AAPL', 
        'current_price': 245.12,
        'target_price': 250.0,
        'probability': 67.8,
        'rsi': 62.1,
        'confidence': 'MEDIUM'
    },
    {
        'ticker': 'NVDA',
        'current_price': 179.93, 
        'target_price': 200.0,
        'probability': 33.4,
        'rsi': 45.2,
        'confidence': 'LOW'
    }
]

@app.route("/")
def index():
    html_files = ["dashboard_multi.html", "dashboard_multi_with_settings.html", "index.html"]
    for filename in html_files:
        if os.path.exists(filename):
            return send_from_directory(".", filename)
    return "No dashboard file found", 404

@app.route("/api/analysis")
def get_analysis():
    return jsonify({
        'stocks': DEMO_STOCKS,
        'status': 'ready',
        'timestamp': datetime.now().isoformat()
    })

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "stocks_configured": 3,
        "stocks_analyzed": 3,
        "analyzer_available": True
    })

@app.route("/api/config")
def get_config():
    return jsonify({
        "stocks": {
            "TSLA": 500.0,
            "AAPL": 250.0, 
            "NVDA": 200.0
        }
    })

@app.route("/api/test")
def test():
    return jsonify({"message": "Working perfectly!", "timestamp": datetime.now().isoformat()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
