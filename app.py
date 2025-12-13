from flask import Flask, request, render_template_string
import ccxt
import pandas as pd

# ---------------- SETTINGS ----------------
ATR_MULTIPLIER = 1.5
TIMEFRAME = "4h"
LIMIT = 60
# ------------------------------------------

app = Flask(__name__)

exchange = ccxt.okx({
    "enableRateLimit": True
})
exchange.load_markets()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trade Evaluator</title>
</head>
<body>
    <h2>Trade Evaluator</h2>
    <form method="POST">
        <input name="ticker" placeholder="BTC, ETH, ADA" value="{{ ticker or '' }}" required>
        <button type="submit">Evaluate</button>
    </form>
    <hr>
    {% if result %}
    <h3>Results for {{ ticker }}</h3>
    <div>{{ result|safe }}</div>
{% endif %}

</body>
</html>
"""

# ---------------- DATA ENGINE ----------------
def get_trend(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    # Price indicators
    df["sma30"] = df["close"].rolling(30).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    # Volume
    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # ATR
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = (
        df[["high", "prev_close"]].max(axis=1)
        - df[["low", "prev_close"]].min(axis=1)
    )
    df["atr14"] = df["tr"].rolling(14).mean()

    sma30 = df["sma30"].iloc[-1]
    sma50 = df["sma50"].iloc[-1]
    price = df["close"].iloc[-1]

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["vol_avg20"].iloc[-1]
    atr = df["atr14"].iloc[-1]

    distance = abs(sma30 - sma50) / price

    if sma30 > sma50:
        direction = "Bullish"
    elif sma30 < sma50:
        direction = "Bearish"
    else:
        direction = "No Trend"

    return {
        "direction": direction,
        "distance": distance,
        "current_volume": current_volume,
        "avg_volume": avg_volume,
        "atr": atr,
        "price": price
    }

# ---------------- APP LOGIC ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ticker = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()

        try:
            symbol = f"{ticker}/USDT"

            if exchange.symbols is None or symbol not in exchange.symbols:
                result = "Invalid symbol."
            else:
                data = get_trend(symbol)

                direction = data["direction"]
                distance = data["distance"]
                price = data["price"]
                atr = data["atr"]

                volume_ratio = (
                    data["current_volume"] / data["avg_volume"]
                    if data["avg_volume"]
                    else 0
                )

                # ---- Regime ----
                if distance >= 0.01:
                    regime = "Expansion"
                elif distance >= 0.005:
                    regime = "Healthy"
                elif distance >= 0.002:
                    regime = "Weak"
                else:
                    regime = "Chop"

                # ---- TQI ----
                tqi = 0

                if direction in ["Bullish", "Bearish"]:
                    tqi += 40

                if regime == "Expansion":
                    tqi += 40
                elif regime == "Healthy":
                    tqi += 30
                elif regime == "Weak":
                    tqi += 15

                if regime in ["Expansion", "Healthy"]:
                    tqi += 20

                # ---- Volume ----
                if volume_ratio >= 1.5:
                    volume_state = "Strong"
                elif volume_ratio >= 1.2:
                    volume_state = "Moderate"
                else:
                    volume_state = "Weak"

                # ---- Final Grade ----
                if volume_state == "Weak":
                    grade = "Watch (No Volume)"
                elif tqi >= 85:
                    grade = "A (High Conviction)"
                elif tqi >= 70:
                    grade = "B (Reduced Size)"
                else:
                    grade = "Skip"

                # ---- Execution ----
                if grade.startswith("A") or grade.startswith("B"):
                    risk = atr * ATR_MULTIPLIER

                    if direction == "Bullish":
                        entry = price
                        stop = entry - risk
                        tp1 = entry + risk
                        tp2 = entry + risk * 2
                        tp3 = entry + risk * 3
                    else:
                        entry = price
                        stop = entry + risk
                        tp1 = entry - risk
                        tp2 = entry - risk * 2
                        tp3 = entry - risk * 3
                else:
                    entry = stop = tp1 = tp2 = tp3 = "N/A"

                result = (
                    f"Trend: {direction} ({regime})<br>"
                    f"Volume: {volume_state} ({volume_ratio:.2f}x avg)<br>"
                    f"TQI: {tqi} / 100 ({grade})<br><br>"
                    f"Entry: {entry}<br>"
                    f"Stop Loss: {stop}<br>"
                    f"TP1 (1R): {tp1}<br>"
                    f"TP2 (2R): {tp2}<br>"
                    f"TP3 (3R): {tp3}<br>"
                )

        except Exception as e:
            result = f"ERROR: {e}"

    return render_template_string(HTML, result=result, ticker=ticker)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



   
