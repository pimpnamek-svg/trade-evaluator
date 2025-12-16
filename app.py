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
def home():
    symbol = request.args.get("symbol", "BTC").upper()
    
    if symbol:
        try:
            # Call your working /eval
            import urllib.parse
            eval_url = f"/eval?symbol={urllib.parse.quote(symbol)}"
            # Simulate calling it (use your real logic here)
            result = evaluate(symbol, 0, 0, 0)  # Your working function
        except:
            result = {"error": "Analysis failed"}
    else:
        result = None
        
    return f'''
<!DOCTYPE html>
<html><head><title>Trading Signals</title>
<style>
body {{font-family:Arial;padding:50px;max-width:800px;margin:auto;}}
input {{padding:12px;width:200px;font-size:18px;}}
button {{padding:15px 30px;background:#28a745;color:white;border:none;font-size:18px;cursor:pointer;border-radius:5px;}}
.result {{margin-top:30px;padding:30px;border:2px solid #007bff;border-radius:15px;background:#f8f9fa;}}
.price {{font-size:24px;font-weight:bold;color:#007bff;margin:20px 0;}}
.levels {{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0;}}
.level {{padding:15px;background:white;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);}}
</style>
</head><body>
<h1>🚀 Crypto Trading Analyzer</h1>

<form method="GET">
<label>Symbol: </label>
<input name="symbol" value="{symbol}" placeholder="BTC, ETH, SOL, XRP">
<button>Analyze ➡️</button>
</form>

{""
if not result else f'''
<div class="result">
<h2>📊 {symbol} Analysis</h2>
<div class="price">Current Price: ${result.get("current_price", 0):.2f}</div>

{"<p style='color:orange;font-size:18px;'>{result['error']}</p>" if result.get('error') else ""}

<div class="levels">
<div class="level"><strong>Suggested Entry</strong><br>${result.get("suggested_entry", 0):.2f}</div>
<div class="level"><strong>Suggested Stop</strong><br>${result.get("suggested_stop", 0):.2f}</div>
<div class="level"><strong>Suggested Target</strong><br>${result.get("suggested_target", 0):.2f}</div>
</div>
</div>
'''}
</body></html>
'''


@app.route("/eval")

def evaluate(symbol, entry, stop, target):
    try:
        closes, highs, lows, volumes = provider.get_candles(symbol)
        price = provider.get_current_price(symbol)
        
        ema_fast = ema(closes, 9)[-1]
        ema_slow = ema(closes, 21)[-1]
        rsi_val = rsi(closes)
        atr_val = atr(highs, lows, closes)
        
        # 70% criteria
        sma30 = sum(closes[-30:]) / 30
        sma50 = sum(closes[-50:]) / 50
        bullish_trend = sma30 > sma50
        avg_volume = sum(volumes[-20:]) / 20
        volume_spike = volumes[-1] >= avg_volume * 1.5
        
        # Safety check
        if abs(price - entry) / price > 0.05:
            return {
                "error": "Entry too far from current price",
                "current_price": round(price, 2),
                "suggested_entry": round(ema_fast, 2),
                "suggested_stop": round(price - atr_val * 1.5, 2),
                "suggested_target": round(price + atr_val * 3, 2),
                "score": 0,
                "rr": 0,
                "rsi": round(rsi_val, 1),
                "signal": "HOLD",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Risk/reward
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0
        
        # FULL SCORING
        score = 0
        if bullish_trend and price > ema_fast:
            score += 30
        if volume_spike:
            score += 20
        if rr >= 2:
            score += 20
        if 30 <= rsi_val <= 70:
            score += 15
        if abs(price - ema_slow) / price < 0.02:  # Near EMA21
            score += 15
            
        signal = "🔥 STRONG BUY" if score >= 70 else "✅ BUY" if score >= 50 else "HOLD"
        
        # **RETURN ALL FIELDS YOUR DISPLAY EXPECTS**
        return {
            "symbol": f"{symbol}-USDT",
            "current_price": round(price, 2),
            "suggested_entry": round(ema_fast, 2),
            "suggested_stop": round(ema_fast - atr_val * 1.5, 2),
            "suggested_target": round(ema_fast + atr_val * 3, 2),
            "score": score,
            "rr": rr,
            "rsi": round(rsi_val, 1),
            "sma_trend": "BULLISH" if sma30 > sma50 else "BEARISH",
            "volume_spike": volume_spike,
            "signal": signal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "current_price": 0,
            "suggested_entry": 0,
            "suggested_stop": 0,
            "suggested_target": 0,
            "score": 0,
            "rr": 0,
            "rsi": 0,
            "signal": "ERROR",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        
@app.route('/analyze')
def analyze():
    try:
        symbol = request.args.get("symbol", "BTC").upper()
        entry = float(request.args.get("entry", 87000))
        stop = float(request.args.get("stop", 86000))
        target = float(request.args.get("target", 89000))
        
        result = evaluate(symbol, entry, stop, target)
        
        # PASS FORM VALUES BACK TO FORM
        return home(
            result=result,
            form_symbol=symbol,
            form_entry=entry,
            form_stop=stop,
            form_target=target
        )
    except Exception as e:
        return home(
            result={'error': str(e)},
            form_symbol=request.args.get("symbol", "BTC"),
            form_entry=float(request.args.get("entry", 87000)),
            form_stop=float(request.args.get("stop", 86000)),
            form_target=float(request.args.get("target", 89000))
        )

        
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


