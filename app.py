import requests
import time
from flask import Flask, request, jsonify

OKX_BASE = "https://www.okx.com"  # Keep this
# But test these alternatives in get_candles/get_current_price:
# "https://aws.okx.com"           # Primary US/Global
# "https://www.okx.com"           # Fallback


class OKXProvider:
    def __init__(self):
        self.bases = [
            "https://aws.okx.com",
            "https://www.okx.com",
            "https://www.okx.com"
        ]

    def _request(self, url, params):
        for base in self.bases:
            try:
                full_url = f"{base}/api/v5{url}"
                r = requests.get(full_url, params=params, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("code") == "0":
                        return data
            except:
                continue
        raise Exception("All OKX endpoints unreachable")

    def get_current_price(self, symbol: str) -> float:
        inst = self.normalize_symbol(symbol)
        data = self._request("/market/ticker", {"instId": inst})
        return float(data["data"][0]["last"])

    def get_candles(self, symbol: str, limit=200):
    inst = self.normalize_symbol(symbol)
    bases = ["https://aws.okx.com", "https://www.okx.com"]
    
    for base in bases:
        try:
            url = f"{base}/api/v5/market/candles"
            r = requests.get(url, params={
                "instId": inst,
                "bar": "1H",
                "limit": str(limit)
            }, timeout=5).json()
            
            if r.get("code") == "0" and r["data"]:
                data = r["data"][::-1]  # Chronological order
                closes = [float(c[4]) for c in data]
                highs = [float(c[2]) for c in data]
                lows = [float(c[3]) for c in data]
                volumes = [float(c[5]) for c in data]
                return closes, highs, lows, volumes
        except:
            continue
    raise Exception(f"Failed to fetch {inst} candles")



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
        "current_price": round(price, 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "suggested_entry": suggested_entry,
        "suggested_stop": suggested_stop,
        "suggested_target": suggested_target,
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
        "rsi": round(rsi_val, 1),
        "atr": round(atr_val, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr_ratio": rr,
        "score": score,
        "signal": signal,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

   app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

