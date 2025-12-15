import os
import sys
import time
import json
from flask import Flask, request, jsonify
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import threading
from collections import deque

# ============================================================
# CASH REGISTER WHALE ALERTS + FULL TRADING EVALUATOR
# ============================================================

class ToolConfig:
    """Enhanced ToolConfig with whale/alert settings"""
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
        
        # 🐋 CASH REGISTER WHALE SETTINGS
        self.whale_volume_mult = float(os.environ.get("WHALE_VOLUME_MULT", "10.0"))
        self.whale_window = int(os.environ.get("WHALE_WINDOW", "6"))
        
        # 🌙 QUIET ACCUMULATION
        self.accumulation_threshold = float(os.environ.get("ACCUM_THRESHOLD", "0.02"))
        self.quiet_period = int(os.environ.get("QUIET_PERIOD", "24"))
        
        # 📈 TIER PROMOTION
        self.tier_thresholds = {
            "bronze": 60, "silver": 75, "gold": 85, "platinum": 95
        }

class CashRegisterWhaleAlert:
    """💰💰💰 CASH REGISTER SOUND EFFECT"""
    def __init__(self):
        self.enabled = True
    
    def play_cash_register(self, symbol: str, whale_ratio: float):
        """💰💰💰 *CHA-CHING* Whale Alert! 💰💰💰"""
        if not self.enabled:
            return
            
        # CASH REGISTER SOUND EFFECT (text + emoji animation)
        cash_sound = """
💰💰💰💰💰💰💰💰💰💰💰
💵 *CHA-CHING!* 🐋 WHALE DETECTED 🐋
💰 {symbol}: {whale_ratio:.1f}x Volume Spike! 💰
💰💰💰💰💰💰💰💰💰💰💰
        """.format(symbol=symbol, whale_ratio=whale_ratio)
        
        print("\a" + cash_sound + "\a")  # Terminal bell + visual
        sys.stdout.flush()

class AlertManager:
    """🔔 Enhanced alert system"""
    def __init__(self):
        self.alerts = deque(maxlen=100)
        self.discord_webhook = os.environ.get("DISCORD_WEBHOOK")
        self.cash_register = CashRegisterWhaleAlert()
    
    def send_whale_alert(self, symbol: str, whale_data: Dict):
        """🐋 Special WHALE alert with cash register sound"""
        self.cash_register.play_cash_register(symbol, whale_data["whale_ratio"])
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "title": f"🐋💰 WHALE ACTIVITY DETECTED 💰🐋",
            "symbol": symbol,
            "whale_ratio": whale_data["whale_ratio"],
            "confidence": whale_data["confidence"],
            "priority": "CRITICAL"
        }
        self.alerts.append(alert)
        
        print(f"🔔 CRITICAL WHALE ALERT 🔔 {symbol}: {whale_data['whale_ratio']}x volume!")
        
        # Discord (if configured)
        if self.discord_webhook:
            try:
                import requests
                requests.post(self.discord_webhook, json={
                    "content": f"💰💰 *CHA-CHING!* 🐋 **{symbol} WHALE ALERT** 🐋\n{whale_data['whale_ratio']}x volume spike!"
                })
            except:
                pass

    def send_regular_alert(self, title: str, message: str, priority: str = "medium"):
        alert = {"timestamp": datetime.now().isoformat(), "title": title, "message": message, "priority": priority}
        self.alerts.append(alert)
        print(f"🔔 [{priority.upper()}] {title}: {message}")

class DataProvider:
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.tool_cfg = None
        self.start_time = 0
        self.last_alerts = {}
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list:
        now = time.time()
        ohlcv = []
        for i in range(limit):
            t = now - i * 3600
            price = 42500 + (i % 10) * 50
            volume = 1000 + (i % 5) * 200
            
            # 🐋 Simulate MASSIVE whale activity
            if i % 13 == 0:  # Frequent whales for demo
                volume *= 15  # 15x spike!
                price += 300  # Big pump
                
            ohlcv.append([t, price, price + 50, price - 20, price, volume])
        
        self.price_history[symbol] = [c[1] for c in ohlcv]
        self.volume_history[symbol] = [c[5] for c in ohlcv]
        return ohlcv
    
    def get_current_price(self, symbol: str) -> float:
        return 42800.0  # Whale-pumped price

provider = DataProvider()
alerts = AlertManager()

def calculate_rsi(prices: list, period: int = 14) -> list:
    if len(prices) < period + 1: return [50] * len(prices)
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [max(-d, 0) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi = [50] * period
    for i in range(period, len(prices)):
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi_val = 100 - (100 / (1 + rs))
        rsi.append(rsi_val)
    return rsi

def calculate_ema(prices: list, period: int) -> list:
    if not prices: return []
    multiplier = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema_val = (price * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_val)
    return ema

def detect_whale_activity(symbol: str, volumes: list, window: int, tool_cfg) -> Dict[str, Any]:
    """🐋 WHALE DETECTION with cash register trigger"""
    if len(volumes) < window: 
        return {"whale_detected": False, "confidence": 0}
    
    recent_volumes = volumes[-window:]
    avg_volume = sum(recent_volumes[:-3]) / max(1, len(recent_volumes) - 3)
    peak_volume = max(recent_volumes[-3:])
    
    whale_ratio = peak_volume / avg_volume if avg_volume > 0 else 0
    confidence = min(100, max(0, (whale_ratio - 5) * 20))
    
    whale_detected = whale_ratio > tool_cfg.whale_volume_mult
    
    # 💰 TRIGGER CASH REGISTER ON WHALE!
    if whale_detected:
        alerts.send_whale_alert(symbol, {
            "whale_ratio": round(whale_ratio, 1),
            "confidence": round(confidence, 1),
            "peak_volume": peak_volume,
            "avg_volume": round(avg_volume, 0)
        })
    
    return {
        "whale_detected": whale_detected,
        "whale_ratio": round(whale_ratio, 1),
        "confidence": round(confidence, 1),
        "peak_volume": peak_volume,
        "avg_volume": round(avg_volume, 0)
    }

def detect_quiet_accumulation(symbol: str, prices: list, volumes: list, tool_cfg) -> Dict[str, Any]:
    if len(prices) < 24: return {"accumulation": False, "strength": 0}
    start_price = prices[-24]
    end_price = prices[-1]
    price_change = (end_price - start_price) / start_price
    avg_volume = sum(volumes[-24:]) / 24
    low_volume_period = sum(1 for v in volumes[-12:] if v < avg_volume * 0.7)
    accumulation = price_change > tool_cfg.accumulation_threshold and low_volume_period >= 8
    return {
        "accumulation": accumulation,
        "price_change_pct": round(price_change * 100, 2),
        "low_volume_candles": low_volume_period,
        "strength": min(100, abs(price_change * 1000) + low_volume_period * 5)
    }

def get_tier(score: float, tool_cfg) -> str:
    thresholds = tool_cfg.tier_thresholds
    if score >= thresholds["platinum"]: return "🏆 PLATINUM"
    elif score >= thresholds["gold"]: return "🥇 GOLD" 
    elif score >= thresholds["silver"]: return "🥈 SILVER"
    elif score >= thresholds["bronze"]: return "🥉 BRONZE"
    return "⚪ UNRATED"

def evaluate_symbol(provider, symbol, tool_cfg, entry, stop, target):
    ohlcv = provider.get_ohlcv(symbol, tool_cfg.timeframe, tool_cfg.candles)
    closes = [c[1] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]
    current_price = provider.get_current_price(symbol)
    
    rsi = calculate_rsi(closes)
    ema_fast = calculate_ema(closes, tool_cfg.ema_fast)
    ema_slow = calculate_ema(closes, tool_cfg.ema_slow)
    
    whale_data = detect_whale_activity(symbol, volumes, tool_cfg.whale_window, tool_cfg)
    accum_data = detect_quiet_accumulation(symbol, closes, volumes, tool_cfg)
    
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    
    score = 0
    criteria = {
        "price_above_ema_fast": current_price > ema_fast[-1],
        "ema_fast_above_ema_slow": ema_fast[-1] > ema_slow[-1],
        "rsi_oversold": rsi[-1] < tool_cfg.rsi_oversold,
        "volume_above_avg": volumes[-1] > sum(volumes[-20:])/20 * tool_cfg.volume_threshold,
        "risk_reward_ok": rr_ratio >= tool_cfg.risk_reward_min,
        "whale_activity": whale_data["whale_detected"],
        "quiet_accumulation": accum_data["accumulation"]
    }
    
    if criteria["price_above_ema_fast"]: score += 20
    if criteria["ema_fast_above_ema_slow"]: score += 20
    if criteria["rsi_oversold"]: score += 15
    if criteria["volume_above_avg"]: score += 15
    if criteria["risk_reward_ok"]: score += 20
    if whale_data["whale_detected"]: score += 25
    if accum_data["accumulation"]: score += 15
    
    tier = get_tier(score, tool_cfg)
    signal = "HOLD"
    if score >= 85: signal = "🚀 STRONG BUY"
    elif score >= 70: signal = "✅ BUY"
    elif score >= 55: signal = "➡️ WEAK BUY"
    
    return {
        "symbol": symbol, "current_price": round(current_price, 2),
        "entry": entry, "stop": stop, "target": target,
        "risk": round(risk, 2), "reward": round(reward, 2), "rr_ratio": round(rr_ratio, 2),
        "rsi": round(rsi[-1], 1), "score": score, "tier": tier, "signal": signal,
        "whale": whale_data, "accumulation": accum_data, "criteria": criteria,
        "timestamp": datetime.now().isoformat()
    }

def auto_scout_loop():
    tool_cfg = provider.tool_cfg
    symbols = ["BTC", "ETH", "SOL", "SPY", "QQQ", "AAPL"]
    while True:
        for symbol in symbols:
            result = evaluate_symbol(provider, symbol, tool_cfg, 42500, 41200, 46500)
            if result["score"] >= 70:
                print(f"🤖 {symbol}: {result['signal']} ({result['score']}) 🐋{result['whale']['whale_ratio']}x")
        time.sleep(300)

def create_app():
    provider.tool_cfg = ToolConfig()
    app = Flask(__name__)
    
    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "🚀 CASH REGISTER WHALE EVALUATOR LIVE 💰💰💰",
            "features": ["🐋CashRegisterWhales", "🌙Accum", "📈Tiers", "🔔Alerts", "🤖AutoScout"],
            "endpoints": ["/eval", "/scan", "/alerts"],
            "recent_alerts": list(alerts.alerts)[-3:]
        })
    
    @app.route("/eval", methods=["GET"])
    def eval_route():
        symbol = request.args.get("symbol", "").upper().strip()
        try:
            entry = float(request.args.get("entry", 0))
            stop = float(request.args.get("stop", 0))
            target = float(request.args.get("target", 0))
        except: return jsonify({"error": "Numbers required"}), 400
        
        if not symbol or entry <= 0 or target <= 0:
            return jsonify({"error": "Usage: /eval?symbol=BTC&entry=42500&stop=41200&target=46500"}), 400
        
        res = evaluate_symbol(provider, symbol, provider.tool_cfg, entry, stop, target)
        return jsonify(res)
    
    @app.route("/scan", methods=["GET"])
    def scan_all():
        symbols = request.args.get("symbols", "BTC,ETH,SOL").split(",")
        results = []
        for s in symbols:
            if s.strip(): 
                results.append(evaluate_symbol(provider, s.strip().upper(), provider.tool_cfg, 42500, 41200, 46500))
        return jsonify({"results": results})
    
    @app.route("/alerts", methods=["GET"])
    def get_alerts():
        return jsonify({"alerts": list(alerts.alerts)})
    
    return app

if __name__ == "__main__":
    provider.tool_cfg = ToolConfig()
    provider.start_time = time.time()
    provider.last_alerts = {}
    
    run_mode = os.environ.get("RUN_MODE", "server").lower()
    
    if run_mode == "worker":
        print("🤖 Starting AUTO-SCOUT WORKER...")
        auto_scout_loop()
    else:
        print("🚀 Starting CASH REGISTER WHALE SERVER 💰💰💰")
        app = create_app()
        scout_thread = threading.Thread(target=auto_scout_loop, daemon=True)
        scout_thread.start()
        
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port, debug=False)


        



   


  

