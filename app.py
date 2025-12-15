def create_app():
    app = Flask(__name__)
    tool_cfg = ToolConfig()
    
    @app.route("/", methods=["GET"])
    def home():
        import requests
        import time
        
        def get_live_price(symbol):
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
                resp = requests.get(url, timeout=5).json()
                return resp[symbol.lower()]['usd']
            except:
                defaults = {'BTC': 95000, 'ETH': 3800, 'SOL': 220}
                return defaults.get(symbol, 95000)
        
        action = request.args.get("action", "scan")
        
        if action == "scan":
            symbols = ["BTC", "ETH", "SOL"]
            results = []
            
            for symbol in symbols:
                try:
                    live_price = get_live_price(symbol)
                    test_entry = live_price * 0.995
                    test_stop = test_entry * 0.97
                    test_target = test_entry * 1.10
                    
                    provider.refresh_data()
                    time.sleep(0.1)
                    
                    result = evaluate_symbol(provider, symbol, tool_cfg, test_entry, test_stop, test_target)
                    
                    results.append({
                        "symbol": symbol,
                        "live_price": live_price,
                        "suggested_entry": test_entry,
                        "signal": result['signal'],
                        "score": result['score'],
                        "rsi": result['rsi']
                    })
                except:
                    pass
            
            top_signals = [r for r in results if r['score'] > 60]
            
            if top_signals:
                html_results = ""
                for r in top_signals:
                    html_results += f'<div style="background:#333;padding:15px;margin:10px;border-radius:10px"><strong>{r["symbol"]}</strong> ${r["live_price"]:,.0f} → Entry ${r["suggested_entry"]:,.0f} <strong>{r["signal"]} ({r["score"]}/100)</strong> RSI:{r["rsi"]:.0f}</div>'
            else:
                html_results = '<p>No strong signals right now...</p>'
            
            return f"""
            <html><body style='font-family:Arial;background:#1a1a1a;color:white;padding:50px;max-width:800px;margin:auto'>
                <h1>🚀 WHALE ENTRY SCANNER</h1>
                <h2>Live signals (Score > 60)</h2>
                {html_results}
                <form method="GET"><input type="hidden" name="action" value="scan"><button style="padding:20px 50px;background:#4CAF50;color:white;border:none;font-size:20px;border-radius:10px">SCAN ➡️</button></form>
            </body></html>
            """
        
        return f"""
        <html><body style='font-family:Arial;background:#1a1a1a;color:white;padding:50px;max-width:600px;margin:auto'>
            <h1>🚀 CASH REGISTER</h1>
            <form method="GET"><input type="hidden" name="action" value="scan"><button style="padding:20px 50px;background:#4CAF50;color:white;border:none;font-size:20px;border-radius:10px">FIND ENTRIES ➡️</button></form>
        </body></html>
        """
    
    return app, tool_cfg



        



   


  

