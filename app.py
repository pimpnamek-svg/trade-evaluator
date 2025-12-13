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
    
    distance = abs(df["sma30"].iloc[-1] - df["sma50"].iloc[-1]) / df["close"].iloc[-1]

    if df["sma30"].iloc[-1] > df["sma50"].iloc[-1]:
        if distance > 0.01:
            return "Bullish (Strong)"
        else:
            return "Bullish (Weak)"
    elif df["sma30"].iloc[-1] < df["sma50"].iloc[-1]:
        if distance > 0.01:
            return "Bearish (Strong)"
        else:
            return "Bearish (Weak)"
    else:
        return "No Trend"


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
                trend = "Pair not available on OKX"
            else:
                trend = get_trend(symbol)

        except Exception as e:
            trend = f"Error: {str(e)}"

        result = (
            f"Trend: {trend}<br>"
            "Trade Quality Index: TBD<br>"
            "Stop Loss: TBD<br>"
            "TP1 (1R): TBD<br>"
            "TP2 (2R): TBD<br>"
            "TP3 (3R): TBD"
        )

    return render_template_string(HTML, result=result, ticker=ticker)

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


   
