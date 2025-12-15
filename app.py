import os
import sys
from flask import Flask, request, jsonify
from typing import Dict, Any
import time

# ============================================================
# COMPLETE TRADING EVALUATOR - READY FOR RAILWAY
# ============================================================

class ToolConfig:
    """Complete ToolConfig with all trading parameters"""
    def __init__(self):
        self.market = os.environ.get("MARKET", "crypto")
        self.timeframe = os.environ.get("TIMEFRAME", "1h")
        self.candles = int(os.environ.get("CANDLES", "200"))
        self.risk_reward_min = float(os.environ.get("RISK_REWARD_MIN", "2.0"))
        self.volume_threshold = float(os.environ.get("VOLUME_THRESHOLD", "1.5"))
        self.rsi_oversold = int(os.environ.get("RSI_OVERSOLD", "30"))
        self.rsi_overbought = int(os.environ.get("RSI_OVERBOUGHT", "70"))
        self.ema_fast = int(os.environ.get("EMA_FAST", "9"))
        self.ema_slow = int(os.environ.get("EMA_SLOW", "21"))
        self.max_risk_percent = float(os.environ.get("MAX_RISK_PERCENT", "2.0"))

class DataProvider:
    """Mock data provider - replace with yfinance, ccxt, etc."""
    def __init__(self):
        self.cache = {}
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list:
        """Returns mock OHLCV data for testing"""
        # In production: use yfinance, ccxt, or your real data source
        return [
            [time.time() - i*3600, 42500 + i*10, 42600 + i*10, 42400 + i*10, 42550 + i*10, 1000 + i*100]
            for i in range(limit)
        ]
    
    def get_current_price(self, symbol: str) -> float:
        """Returns current price"""
        return 42500.0  # Mock price

provider = DataProvider()

def calculate_rsi(prices: list, period: int = 14) -> list:
    """Complete RSI calculation"""
    if len(prices) < period + 1:
        return [50] * len(prices)
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    rsi = []
    for i in range(len(prices)):
        if i < period:
            rsi.append(50)
        else:
            current_gain = gains[-1] if len(gains) > 0 else 0
            current_loss = losses[-1] if len(losses) > 0 else 0
            
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
            
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi_val = 100 - (100 / (1 + rs))
            rsi.append(rsi_val)
    
    return rsi

def calculate_ema(prices: list, period: int) -> list:
    """Complete EMA calculation"""
    if len(prices) == 0:
        return []
    
    multiplier = 2 / (period + 1)
    ema = [prices[0]]
    
    for price in prices[1:]:
        ema_val = (price * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_val)
    
    return ema

def detect_quiet_accumulation(symbol: str, prices: list, volumes: list, tool_cfg: ToolConfig) -> Dict[str, Any]:
    """
    Quiet accumulation detector with your two tweaks:
      1) If price_change_pct < 0.2 → accumulation = True
      2) If low_volume_period >= 12 and strength > 55 → accumulation = True
    """
    if len(prices) < 24:
        return {"accumulation": False, "strength": 0}
    
    start_price = prices[-24]
    end_price = prices[-1]
    price_change = (end_price - start_price) / start_price
    price_change_pct = price_change * 100
    avg_volume = sum(volumes[-24:]) / 24
    low_volume_period = sum(1 for v in volumes[-12:] if v < avg_volume * 0.7)
    
    strength = min(100, abs(price_change * 1000) + low_volume_period * 5)
    
    accumulation = price_change > tool_cfg.accumulation_threshold if hasattr(tool_cfg, "accumulation_threshold") else False
    if low_volume_period >= 8:
        accumulation = True
    
    if price_change_pct < 0.2:
        accumulation = True
    if low_volume_period >= 12 and strength > 55:
        accumulation = True
    
    return {
        "accumulation": accumulation,
        "price_change_pct": round(price_change_pct, 2),
        "low_volume_candles": low_volume_period,
        "strength": round(strength, 1),
    }

def evaluate_symbol(
    provider: DataProvider,
    symbol: str,
    tool_cfg: ToolConfig,
    entry: float,
    stop: float,
    target: float
) -> Dict[str, Any]:
    """Main evaluation logic for one symbol"""
    
    ohlcv = provider.get_ohlcv(symbol, tool_cfg.timeframe, tool_cfg.candles)
    closes = [candle[1] for candle in ohlcv]
    volumes = [candle[5] for candle in ohlcv]
    current_price = provider.get_current_price(symbol)
    
    rsi = calculate_rsi(closes)
    ema_fast = calculate_ema(closes, tool_cfg.ema_fast)
    ema_slow = calculate_ema(closes, tool_cfg.ema_slow)
    
    current_rsi = rsi[-1] if rsi else 50
    current_ema_fast = ema_fast[-1] if ema_fast else closes[-1]
    current_ema_slow = ema_slow[-1] if ema_slow else closes[-1]
    avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
    current_volume = volumes[-1] if volumes else 0
    
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    
    criteria = {
        "price_above_ema_fast": current_price > current_ema_fast,
        "ema_fast_above_ema_slow": current_ema_fast > current_ema_slow,
        "rsi_oversold": current_rsi < tool_cfg.rsi_oversold,
        "volume_above_avg": current_volume > avg_volume * tool_cfg.volume_threshold,
        "risk_reward_ok": rr_ratio >= tool_cfg.risk_reward_min,
        "position_size_ok": (risk / entry * 100) <= tool_cfg.max_risk_percent if entry > 0 else False,
    }
    
    score = 0
    if criteria["price_above_ema_fast"]: score += 20
    if criteria["ema_fast_above_ema_slow"]: score += 20
    if criteria["rsi_oversold"]: score += 15
    if criteria["volume_above_avg"]: score += 15
    if criteria["risk_reward_ok"]: score += 20
    if criteria["position_size_ok"]: score += 10
    
    signal = "HOLD"
    if score >= 80:
        signal = "STRONG BUY"
    elif score >= 60:
        signal = "BUY"
    elif score >= 40:
        signal = "WEAK BUY"
    elif score <= 20:
        signal = "STRONG SELL"
    
    accum = detect_quiet_accumulation(symbol, closes, volumes, tool_cfg)
    
    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr_ratio": round(rr_ratio, 2),
        "rsi": round(current_rsi, 1),
        "score": score,
        "signal": signal,
        "criteria": criteria,
        "accumulation": accum,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def cli_main(tool_cfg: ToolConfig):
    """Worker mode - continuous scanning"""
    print(f"Worker started: {vars(tool_cfg)}")
    symbols = ["BTC", "ETH", "SPY", "QQQ"]
    
    while True:
        results = []
        for symbol in symbols:
            try:
                result = evaluate_symbol(provider, symbol, tool_cfg, 
                                        entry=42500, stop=41200, target=46500)
                results.append(result)
                print(f"{symbol}: {result['signal']} (Score: {result['score']})")
            except Exception as e:
                print(f"Error evaluating {symbol}: {e}")
        
        print(f"Scan complete: {len([r for r in results if r['score'] >= 60])} BUY signals")
        time.sleep(60)

# ============================================================
# FLASK APP FACTORY (RAILWAY READY)
# ============================================================
def create_app():
    app = Flask(__name__)
    tool_cfg = ToolConfig()
    
    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "Trade Evaluator LIVE",
            "endpoints": {
                "eval": "/eval?symbol=BTC&entry=42500&stop=41200&target=46500",
                "scan": "/scan?symbols=BTC,ETH,SPY,QQQ",
            },
            "config": vars(tool_cfg),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    @app.route("/eval", methods=["GET"])
    def eval_route():
        symbol = request.args.get("symbol", "").upper().strip()
        try:
            entry = float(request.args.get("entry", 0))
            stop = float(request.args.get("stop", 0))
            target = float(request.args.get("target", 0))
        except (ValueError, TypeError):
            return jsonify({"error": "entry/stop/target must be numbers"}), 400
        
        if not symbol or entry <= 0 or stop <= 0 or target <= 0:
            return jsonify({
                "error": "Usage: /eval?symbol=BTC&entry=42500&stop=41200&target=46500"
            }), 400
        
        try:
            res = evaluate_symbol(provider, symbol, tool_cfg, entry, stop, target)
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/scan", methods=["GET"])
    def scan_all():
        symbols = request.args.get("symbols", "BTC,ETH,SPY,QQQ").upper().split(",")
        results = []
        
        for symbol in symbols:
            symbol = symbol.strip()
            if symbol:
                try:
                    res = evaluate_symbol(
                        provider,
                        symbol,
                        tool_cfg,
                        entry=42500,
                        stop=41200,
                        target=46500,
                    )
                    results.append(res)
                except Exception as e:
                    results.append({"symbol": symbol, "error": str(e)})
        
        return jsonify({
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return app, tool_cfg

# ============================================================
# RAILWAY ENTRY POINT
# ============================================================
if __name__ == "__main__":
    tool_cfg = ToolConfig()
    run_mode = os.environ.get("RUN_MODE", "server").lower()
    
    if run_mode == "worker":
        print("Starting WORKER mode...")
        cli_main(tool_cfg)
    else:
        print("Starting SERVER mode...")
        app, tool_cfg = create_app()
        port = int(os.environ.get("PORT", 8080))
        host = os.environ.get("HOST", "0.0.0.0")
        print(f"Trade Evaluator running on http://{host}:{port}")
        print("Endpoints: /, /eval, /scan")
        print(f"Config: {vars(tool_cfg)}")
        app.run(host=host, port=port, debug=False)


        



   


  

