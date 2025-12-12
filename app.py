from flask import Flask
import yfinance as yf 
import pandas as pd
import requests

app = Flask(__name__)

def run_scan():
    return "scan complete."
@app.route("/")
def home():
    result = run_scan()
    return f"✅ Trade Evaluator is LIVE.<br>{result}"

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8080)
