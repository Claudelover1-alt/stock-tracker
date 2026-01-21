from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    html_files = ["dashboard
