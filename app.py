import requests
import time
from flask import Flask, request, jsonify, render_template  # Add render_template

OKX_BASE = "https://www.okx.com"  # Keep this
# But test these alternatives in get_candles/get_current_price:
# "https://aws.okx.com"           # Primary US/Global
# "https://www.okx.com"           # Fallback


class OKXProvider:
    def __init__(self):
        self.bases = ["https://aws.okx.com", "https://www.okx.com"]

    def normalize_symbol(self, symbol: str) -> str:
        return f"{symbol.upper()}-USDT"

    def _request(self, endpoint, params):
        import requests
        for base in self.bases:
            try:
                url = f"{base}/api/v5{endpoint}"
                r = requests.get(url, params=params, timeout=5)
                r.raise_for_status()
                data = r.json()
                if data.get("code") == "0":
                    return data
            except Exception:
                continue
        raise Exception("All OKX endpoints failed")

    def get_current_price(self, symbol: str) -> float:
        data = self._request("/market/ticker", {
            "instId": self.normalize_symbol(symbol)
        })
        return float(data["data"][0]["last"])

    def get_candles(self, symbol: str, limit=200):
        data = self._request("/market/candles", {
            "instId": self.normalize_symbol(symbol),
            "bar": "1H",
            "limit": str(limit),
        })
        candles = data["data"][::-1]
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        volumes = [float(c[5]) for c in candles]
        return closes, highs, lows, volumes


provider = OKXProvider()

def ema(prices, period):
    if len(prices) < period:
        return [prices[-1]] * len(prices)
    k = 2 / (period + 1)
    e = [prices[0]]
    for p in prices[1:]:
        e.append(p * k + e[-1] * (1 - k))
    return e  # Add this line

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(abs(min(d, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period or 1
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.01
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    return sum(trs[-period:]) / period


def evaluate(symbol, entry, stop, target):
    try:
        closes, highs, lows, volumes = provider.get_candles(symbol)
        price = provider.get_current_price(symbol)
        
        # Calculate indicators INSIDE try block
        ema_fast = ema(closes, 9)[-1]
        ema_slow = ema(closes, 21)[-1]
        rsi_val = rsi(closes)
        atr_val = atr(highs, lows, closes)
        
    except Exception as e:
        return {"error": f"API Error: {str(e)}"}
    
    # Guard: Now all variables are defined
    if abs(price - entry) / price > 0.05:
         return {
            "error": "Entry too far from current price",
            "current_price": round(price, 2),
            "suggested_entry": round(ema_fast, 2),
            "suggested_stop": round(price - atr_val * 1.5, 2),
            "suggested_target": round(price + atr_val * 3, 2),
       
            
        }
    # AUTO LEVEL ASSIST
    suggested_entry = round(ema_fast, 2)
    suggested_stop = round(suggested_entry - atr_val * 1.5, 2)
    suggested_target = round(suggested_entry + atr_val * 3, 2)

    # RISK/REWARD
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0

    # SCORING
    score = 0
    bullish = ema_fast > ema_slow
    if bullish and price > ema_fast:
        score += 30
    if not bullish and price < ema_fast:
        score -= 30
    if rsi_val < 30:
        score += 15
    elif rsi_val > 70:
        score -= 15
    if rr >= 2:
        score += 20

    # SIGNAL
    if score >= 60:
        signal = "STRONG BUY"
    elif score >= 40:
        signal = "BUY"
    elif score <= -60:
        signal = "STRONG SELL"
    elif score <= -40:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol": f"{symbol.upper()}-USDT",
        # ... all your fields ...
        "signal": signal,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Flask app (outside evaluate function)
app = Flask(__name__)

@app.route('/')
def home(result=None):
    html = '''
<!DOCTYPE html>
<html><head><title>Trading Signals</title>
<style>body{font-family:Arial;padding:50px;max-width:800px;margin:auto;}
input{padding:12px;margin:5px;font-size:16px;}
button{padding:15px 25px;background:#007bff;color:white;border:none;font-size:16px;cursor:pointer;}
.result{margin-top:30px;padding:25px;border:2px solid #ddd;border-radius:10px;background:#f9f9f9;}
.signal{font-size:24px;font-weight:bold;padding:20px;border-radius:8px;text-align:center;margin:15px 0;}
.buy{background:#d4edda;color:#155724;}
.sell{background:#f8d7da;color:#721c24;}
.hold{background:#fff3cd;color:#856404;}
.error{background:#f8d7da;color:#721c24;}
</style></head><body>
<h1>🚀 Crypto Trading Signals</h1>
<form method="GET" action="/analyze">
Symbol: <input name="symbol" value="BTC" style="width:150px;"><br><br>
Entry: $<input name="entry" value="87000" type="number" step="0.01" style="width:150px;">
Stop: $<input name="stop" value="86000" type="number" step="0.01" style="width:150px;"><br><br>
Target: $<input name="target" value="89000" type="number" step="0.01" style="width:150px;"><br><br>
<button type="submit">Analyze Trade ➡️</button>
</form>
'''

    if result:
        html += f'''
<div class="result">
<h2>📊 Analysis Results</h2>
<div class="signal {result.get('signal', 'hold').lower()}">
Signal: {result.get('signal', 'N/A')}
</div>
<p><strong>Symbol:</strong> {result.get('symbol', 'N/A')}</p>
<p><strong>Current Price:</strong> ${round(result.get('current_price', 0), 2)}</p>
<p><strong>Suggested Entry:</strong> ${round(result.get('suggested_entry', 0), 2)}</p>
<p><strong>Suggested Stop:</strong> ${round(result.get('suggested_stop', 0), 2)}</p>
<p><strong>Suggested Target:</strong> ${round(result.get('suggested_target', 0), 2)}</p>
<p><strong>Score:</strong> {result.get('score', 0)}/100</p>
<p><strong>Risk/Reward:</strong> {result.get('rr', 0)}:1</p>
<p><strong>RSI:</strong> {round(result.get('rsi', 0), 1)}</p>
<p><strong>Timestamp:</strong> {result.get('timestamp', 'N/A')}</p>
{ '<p style="color:red;"><strong>Error:</strong> ' + result.get('error', '') + '</p>' if result.get('error') else '' }
</div>
'''
    
    html += '</body></html>'
    return html


@app.route("/eval")

def eval_route():
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        entry = float(request.args.get("entry", 0))
        stop = float(request.args.get("stop", 0))
        target = float(request.args.get("target", 0))
        return jsonify(evaluate(symbol, entry, stop, target))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
@app.route('/analyze')
def analyze():
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        entry = float(request.args.get("entry", 87000))
        stop = float(request.args.get("stop", 86000))
        target = float(request.args.get("target", 89000))
        
        # Call evaluate with defaults if params invalid
        result = evaluate(symbol, entry, stop, target)
        
        # Ensure result has all keys
        if 'error' not in result:
            result.setdefault('signal', 'HOLD')
            result.setdefault('score', 0)
            result.setdefault('rsi', 50)
            result.setdefault('rr', 0)
        
        return home(result=result)
        
    except Exception as e:
        return home(result={
            'error': f"Error: {str(e)}",
            'signal': 'ERROR',
            'score': 0,
            'rsi': 0,
            'rr': 0,
            'symbol': symbol,
            'current_price': 0,
            'suggested_entry': 0,
            'suggested_stop': 0,
            'suggested_target': 0,
            'timestamp': 'N/A'
        })


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)


