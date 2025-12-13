from flask import Flask, request, render_template_string
import os
import ccxt
import pandas as pd

# --------------------
# App setup
# --------------------
app = Flask(__name__)

# --------------------
# Exchange setup
# --------------------
exchange = ccxt.okx({
    "enableRateLimit": True
})
exchange.load_markets()

# --------------------
# Helper functions
# --------------------
def get_trend(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="4h", limit=60)

    df = pd.DataFrame(
        ohlcv,
        columns=["time", "open", "high", "low", "close", "volume"]
    )

    df["sma30"] = df["close"].rolling(30).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    df["vol_avg20"] = df["volume"].rolling(20).mean()

    sma30 = df["sma30"].iloc[-1]
    sma50 = df["sma50"].iloc[-1]
    price = df["close"].iloc[-1]

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["vol_avg20"].iloc[-1]

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
        "avg_volume": avg_volume
    }

  
# --------------------
# HTML
# --------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trade Evaluator</title>
</head>
<body>
    <h2>Trade Evaluator</h2>

    <form method="POST">
        <input name="ticker" placeholder="BTC, ETH, SOL" required>
        <button type="submit">Evaluate</button>
    </form>

    {% if result %}
        <p>{{ result|safe }}</p>
    {% endif %}
</body>
</html>
"""

# --------------------
# Route
# --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ticker = None

    if request.method == "POST":
        ticker = request.form.get("ticker", "").upper()

        try:
            symbol = f"{ticker}/USDT"

            if exchange.symbols is None or symbol not in exchange.symbols:
                direction = "N/A"
                distance = 0
                regime = "N/A"
                tqi = 0
                grade = "N/A"
                volume_state = "N/A"
                volume_ratio = 0

            else:
                trend_data = get_trend(symbol)

                direction = trend_data["direction"]
                distance = trend_data["distance"]
                current_volume = trend_data["current_volume"]
                avg_volume = trend_data["avg_volume"]

                volume_ratio = current_volume / avg_volume if avg_volume else 0

                # ---- TQI scoring ----
                tqi = 0

                # Trend direction
                if direction in ["Bullish", "Bearish"]:
                    tqi += 40

                # Trend strength
                if distance >= 0.01:
                    tqi += 40
                    regime = "Expansion"
                elif distance >= 0.005:
                    tqi += 30
                    regime = "Healthy"
                elif distance >= 0.002:
                    tqi += 15
                    regime = "Weak"
                else:
                    regime = "Chop"

                # Regime bonus
                if regime in ["Expansion", "Healthy"]:
                    tqi += 20
                elif regime == "Weak":
                    tqi += 10

                # ---- Volume scoring ----
                if volume_ratio >= 1.5:
                    tqi += 20
                    volume_state = "Strong"
                elif volume_ratio >= 1.2:
                    tqi += 10
                    volume_state = "Moderate"
                else:
                    volume_state = "Weak"

                # Grade
                if tqi >= 85:
                    grade = "A (High Conviction)"
                elif tqi >= 70:
                    grade = "B (Reduce Size)"
                else:
                    grade = "Skip"

        except Exception as e:
            direction = "Error"
            distance = 0
            regime = "Error"
            tqi = 0
            grade = str(e)
            volume_state = "Error"
            volume_ratio = 0

        result = (
            f"Trend: {direction} ({regime})<br>"
            f"Distance: {distance:.4%}<br>"
            f"Volume: {volume_state} ({volume_ratio:.2f}x avg)<br>"
            f"Trade Quality Index: {tqi} / 100 ({grade})<br>"
            "Stop Loss: TBD<br>"
            "TP1 (1R): TBD<br>"
            "TP2 (2R): TBD<br>"
            "TP3 (3R): TBD"
        )

    return render_template_string(HTML, result=result, ticker=ticker)


    return render_template_string(HTML, result=result, ticker=ticker)

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


   
