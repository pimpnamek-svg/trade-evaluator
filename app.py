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

    sma30 = df["sma30"].iloc[-1]
    sma50 = df["sma50"].iloc[-1]
    price = df["close"].iloc[-1]

    distance = abs(sma30 - sma50) / price

    if sma30 > sma50:
        direction = "Bullish"
    elif sma30 < sma50:
        direction = "Bearish"
    else:
        direction = "No Trend"

    return {
        "direction": direction,
        "distance": distance
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
            else:
                trend_data = get_trend(symbol)

                direction = trend_data["direction"]
                distance = trend_data["distance"]

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

                # Grade
                if tqi >= 80:
                    grade = "A (Trade)"
                elif tqi >= 65:
                    grade = "B (Trade)"
                elif tqi >= 50:
                    grade = "C (Caution)"
                else:
                    grade = "D (Skip)"

        except Exception as e:
            direction = "Error"
            distance = 0
            regime = "Error"
            tqi = 0
            grade = str(e)

        result = (
            f"Trend: {direction} ({regime})<br>"
            f"Distance: {distance:.4%}<br>"
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


   
