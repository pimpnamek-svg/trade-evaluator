import os
from flask import Flask, request, jsonify
import time
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "🚀 CASH REGISTER WHALE EVALUATOR LIVE 💰💰💰",
        "message": "Whale detection active!"
    })

@app.route("/eval")
def eval_route():
    symbol = request.args.get("symbol", "BTC")
    entry = float(request.args.get("entry", 42500))
    stop = float(request.args.get("stop", 41200))
    target = float(request.args.get("target", 46500))
    
    # Simple whale simulation
    whale_detected = True
    score = 92
    
    return jsonify({
        "symbol": symbol,
        "entry": entry,
        "stop": stop,
        "target": target,
        "whale_detected": whale_detected,
        "score": score,
        "signal": "🚀 STRONG BUY 💰🐋",
        "message": "💰 *CHA-CHING!* Whale activity detected!"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)




        



   


  

