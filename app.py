import requests
import time
import os
from flask import Flask, request

app = Flask(__name__)

OKX_BASE = "https://www.okx.com"
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
        data = self._request("/market/ticker", {"instId": self.normalize_symbol(symbol)})
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
    if len(prices) < period: return [prices[-1]] * len(prices)
    k = 2 / (period + 1)
    e = [prices[0]]
    for p in prices[1:]: e.append(p * k + e[-1] * (1 - k))
    return e

def rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
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
    if len(closes) < period + 1: return 0.01
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs[-period:]) / period

# SINGLE EVALUATE FUNCTION (not a route)
def evaluate(symbol, entry=0, stop=0, target=0):
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
        
        suggested_entry = round(ema_fast, 2)
        suggested_stop = round(ema_fast - atr_val * 1.5, 2)
        suggested_target = round(ema_fast + atr_val * 3, 2)
        
        # Use provided levels if given, else suggested
        final_entry = entry or suggested_entry
        final_stop = stop or suggested_stop
        final_target = target or suggested_target
        
        # Risk/reward
        risk = abs(final_entry - final_stop)
        reward = abs(final_target - final_entry)
        rr = round(reward / risk, 2) if risk > 0 else 0
        
        # Safety check
        if abs(price - final_entry) / price > 0.05:
            return {
                "error": "Entry too far from current price",
                "current_price": round(price, 2),
                "suggested_entry": suggested_entry,
                "suggested_stop": suggested_stop,
                "suggested_target": suggested_target,
                "score": 0, "rr": 0, "rsi": round(rsi_val, 1),
                "sma_trend": "BULLISH" if bullish_trend else "BEARISH",
                "volume_spike": volume_spike,
                "signal": "HOLD",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # FULL SCORING
        score = 0
        if bullish_trend and price > ema_fast: score += 30
        if volume_spike: score += 20
        if rr >= 2: score += 20
        if 30 <= rsi_val <= 70: score += 15
        if abs(price - ema_slow) / price < 0.02: score += 15
        
        signal = "🔥 STRONG BUY" if score >= 70 else "✅ BUY" if score >= 50 else "HOLD"
        
        return {
            "symbol": f"{symbol}-USDT",
            "current_price": round(price, 2),
            "entry": final_entry,
            "stop": final_stop,
            "target": final_target,
            "suggested_entry": suggested_entry,
            "suggested_stop": suggested_stop,
            "suggested_target": suggested_target,
            "score": score,
            "rr": rr,
            "rsi": round(rsi_val, 1),
            "sma_trend": "BULLISH" if bullish_trend else "BEARISH",
            "volume_spike": volume_spike,
            "signal": signal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e), "signal": "ERROR"}

@app.route('/')
def home():
    symbol = request.args.get("symbol", "BTC").upper()
    result = evaluate(symbol) if symbol else None
    
    return f'''
<!DOCTYPE html>
<html><head><title>Trading Signals</title>
<style>body{{font-family:Arial;padding:50px;max-width:900px;margin:auto;}} 
input{{padding:12px;width:200px;}} button{{padding:15px 30px;background:#28a745;color:white;border:none;font-size:16px;cursor:pointer;}}
.result{{background:#f8f9fa;padding:25px;border-radius:10px;margin-top:30px;}} 
.signal{{font-size:28px;font-weight:bold;padding:20px;border-radius:8px;text-align:center;margin:20px 0;}}
.metric{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:20px 0;}}
.item{{padding:15px;background:white;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);}}</style>
</head><body>
<h1>🚀 70% Win Rate Trading Signals</h1>
<form method="GET">
Symbol: <input name="symbol" value="{symbol}">
<button>Analyze ➡️</button>
</form>

{'' if not result else f'''
<div class="result">
<h2>📊 {result.get("symbol", "N/A")}</h2>
<div class="signal">{result.get("signal", "N/A")}</div>
<div class="metric">
<div class="item"><strong>Current</strong><br>${result.get("current_price", 0)}</div>
<div class="item"><strong>Suggested Entry</strong><br>${result.get("suggested_entry", 0)}</div>
<div class="item"><strong>Suggested Stop</strong><br>${result.get("suggested_stop", 0)}</div>
<div class="item"><strong>Suggested Target</strong><br>${result.get("suggested_target", 0)}</div>
<div class="item"><strong>Score</strong><br>{result.get("score", 0)}/100</div>
<div class="item"><strong>R:R</strong><br>{result.get("rr", 0)}:1</div>
<div class="item"><strong>RSI</strong><br>{result.get("rsi", 0)}</div>
<div class="item"><strong>SMA Trend</strong><br>{result.get("sma_trend", "N/A")}</div>
<div class="item"><strong>Volume Spike</strong><br>{result.get("volume_spike", False)}</div>
</div>
{ '<p style="color:red;">' + result.get("error", "") + '</p>' if result.get("error") else '' }
<p><strong>Time:</strong> {result.get("timestamp", "N/A")}</p>
</div>
'''}
</body></html>
'''

@app.route('/eval')
def eval_route():
    symbol = request.args.get("symbol", "BTC").upper()
    entry = float(request.args.get("entry", 0) or 0)
    stop = float(request.args.get("stop", 0) or 0)
    target = float(request.args.get("target", 0) or 0)
    result = evaluate(symbol, entry, stop, target)
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)



