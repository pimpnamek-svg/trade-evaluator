from flask import Flask
import yfinance as yf 
import pandas as pd
import requests

app = Flask(__name__)

from flask import Flask, request, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trade Evaluator</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        input { padding: 10px; font-size: 16px; }
        button { padding: 10px 15px; font-size: 16px; }
        .result { margin-top: 20px; padding: 15px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h2>Trade Evaluator</h2>

    <form method="POST">
        <input name="ticker" placeholder="Enter ticker (BTC, ETH, SOL)" required>
        <button type="submit">Evaluate</button>
    </form>

    {% if result %}
    <div class="result">
        <strong>Result for {{ ticker }}:</strong><br><br>
        {{ result }}
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ticker = None

    import ccxt
import pandas as pd

exchange = ccxt.okx()

def get_trend(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=60)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])

    df['sma30'] = df['close'].rolling(30).mean()
    df['sma50'] = df['close'].rolling(50).mean()

    if df['sma30'].iloc[-1] > df['sma50'].iloc[-1]:
        return "Bullish"
    elif df['sma30'].iloc[-1] < df['sma50'].iloc[-1]:
        return "Bearish"
    else:
        return "No Trend"
     trend = get_trend(f"{ticker}/USDT")

result = (
    f"Trend: {trend}<br>"
    "Trade Quality Index: TBD<br>"
    "Stop Loss: TBD<br>"
    "TP1 (1R): TBD<br>"
    "TP2 (2R): TBD<br>"
    "TP3 (3R): TBD"
)

    return render_template_string(HTML, result=result, ticker=ticker)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

