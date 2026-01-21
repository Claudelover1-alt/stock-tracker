"""
Stock Tracker Server - PERFECT FRONTEND MATCH
Exactly what dashboard_multi.html expects
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

# EXACT data structure your frontend expects
@app.route("/")
def index():
    """Serve dashboard"""
    return send_from_directory(".", "dashboard_multi.html")

@app.route("/api/analysis")
def analysis():
    """Exact frontend format - works instantly"""
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "stocks": [
            {
                "ticker": "TSLA",
                "current_price": 423.35,
                "target_price": 500.0,
                "distance_to_target": 76.65,
                "distance_pct": 18.1,
                "probability": {
                    "composite_probability": 42.3,
                    "momentum_score": 58.2,
                    "statistical_probability": 39.8,
                    "ml_probability": 45.1,
                    "confidence_level": "HIGH"
                },
                "technical_indicators": {
                    "rsi": 58.7,
                    "macd": 2.34,
                    "macd_signal": 1.89,
                    "stoch_k": 67.2,
                    "adx": 28.4
                },
                "status": "ready"
            },
            {
                "ticker": "AAPL",
                "current_price": 245.12,
                "target_price": 250.0,
                "distance_to_target": 4.88,
                "distance_pct": 2.0,
                "probability": {
                    "composite_probability": 67.8,
                    "momentum_score": 72.1,
                    "statistical_probability": 65.3,
                    "ml_probability": 70.2,
                    "confidence_level": "HIGH"
                },
                "technical
