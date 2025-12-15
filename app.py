import requests
import time
from flask import Flask, request, jsonify

OKX_BASE = "https://www.okx.com"

# -------------------------
# OKX DATA PROVIDER
# -------------------------
class OKXProvider:
    def normalize_symbol(self, symbol: str) -> str:
        return f"{symbol}-USDT"

    def get_current_price(self, symbol: str) -> float:
        inst = self.normalize_symbol(symbol)
        url = f"{OKX_BASE}/api/v5/market/ticker"
        r = requests.get(url, params={"instId": inst}, timeout=5).json()
        return float(r["data"][0]["last"])

    def get_candles(self, symbol: str, limit=200):
    inst = self.normalize_symbol(symbol)
    url = f"{OKX_BASE}/api/v5/market/candles"
    r = requests.get(url, params={
        "instId": inst,
        "bar": "1H",
        "limit": limit
    }, timeout=5).json()

    data = r["data"][::-1]

    closes = [float(c[4]) for c in data]
    highs = [float(c[2]) for c in data]
    lows = [float(c[3]) for c in data]
    volumes = [float(c[5]) for c in data]

    return closes, highs, lows, volumes

       
provider = OKXProvider()

# -------------------------
# INDICATORS
# -------------------------
def ema(prices, period):
    k = 2 / (period + 1)
    e = [prices[0]]
    for p in prices[1:]:
        e.append(p * k + e[-1] * (1 - k))
    return e

def rsi(prices, period=14):
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0))
        losses.append(abs(min(d, 0)))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period or 1
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------
# EVALUATOR
# -------------------------

def atr(highs, lows, closes, period=14):
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
    closes, highs, lows, volumes = provider.get_candles(symbol)
    price = provider.get_current_price(symbol)
    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    rsi_val = rsi(closes)
    atr_val = atr(highs, lows, closes)

    # -------------------
    # VALIDATOR GUARD
    # -------------------
    if abs(price - entry) / price > 0.05:
        return {
            "error": "Entry too far from current price",
            "current_price": round(price, 2),
            "suggested_entry": round(price, 2),
            "suggested_stop": round(price - atr_val * 1.5, 2),
            "suggested_target": round(price + atr_val * 3, 2),
        }


    # -------------------
    # AUTO LEVEL ASSIST
    # -------------------
    suggested_entry = round(ema_fast, 2)
    suggested_stop = round(suggested_entry - atr_val * 1.5, 2)
    suggested_target = round(suggested_entry + atr_val * 3, 2)

    # -------------------
    # RISK / REWARD
    # -------------------
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2) if risk else 0

    # -------------------
    # SCORING LOGIC
    # -------------------
    score = 0

    bullish = ema_fast > ema_slow
    bearish = ema_fast < ema_slow

    if bullish and price > ema_fast:
        score += 30
    if bearish and price < ema_fast:
        score -= 30

    if rsi_val < 30:
        score += 15
    elif rsi_val > 70:
        score -= 15

    if rr >= 2:
        score += 20

    # -------------------
    # SIGNAL
    # -------------------
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
        "symbol": f"{symbol}-USDT",
        "current_price": round(price, 2),

        # your inputs
        "entry": entry,
        "stop": stop,
        "target": target,

        # assist
        "suggested_entry": suggested_entry,
        "suggested_stop": suggested_stop,
        "suggested_target": suggested_target,

        # indicators
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
        "rsi": round(rsi_val, 1),
        "atr": round(atr_val, 2),

        # risk
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr_ratio": rr,

        # output
        "score": score,
        "signal": signal,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

@app.route("/eval")
def eval_route():
    symbol = request.args.get("symbol", "").upper()
    entry = float(request.args.get("entry"))
    stop = float(request.args.get("stop"))
    target = float(request.args.get("target"))
    return jsonify(evaluate(symbol, entry, stop, target))

if __name__ == "__main__":
    app.run(port=8080, debug=False)

