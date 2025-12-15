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

    def get_closes_and_volume(self, symbol: str, limit=200):
        inst = self.normalize_symbol(symbol)
        url = f"{OKX_BASE}/api/v5/market/candles"
        r = requests.get(url, params={
            "instId": inst,
            "bar": "1H",
            "limit": limit
        }, timeout=5).json()

        closes = [float(c[4]) for c in r["data"]][::-1]
        volumes = [float(c[5]) for c in r["data"]][::-1]
        return closes, volumes


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
def evaluate(symbol, entry, stop, target):
    closes, volumes = provider.get_closes_and_volume(symbol)
    price = provider.get_current_price(symbol)

    ema_fast = ema(closes, 9)[-1]
    ema_slow = ema(closes, 21)[-1]
    rsi_val = rsi(closes)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2)

    score = 0
    if price > ema_fast: score += 20
    if ema_fast > ema_slow: score += 20
    if rsi_val < 30: score += 15
    if rr >= 2: score += 25

    signal = (
        "STRONG BUY" if score >= 80 else
        "BUY" if score >= 60 else
        "HOLD"
    )

    return {
        "symbol": f"{symbol}-USDT",
        "current_price": round(price, 2),
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr_ratio": rr,
        "rsi": round(rsi_val, 1),
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
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
