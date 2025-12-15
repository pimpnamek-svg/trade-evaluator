# ======================================
# TRADE EVALUATOR TOOL (STABLE REBUILD)
# ======================================

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
import ccxt
from flask import Flask, jsonify, request


# ======================================
# CONFIG
# ======================================

@dataclass
class VolumeTierConfig:
    tier_a_rvol: float = 1.5
    tier_bplus_min_rvol: float = 1.1
    tier_b_min_rvol: float = 0.8


@dataclass
class WhaleFlagConfig:
    atr_fall_lookback: int = 5


@dataclass
class ToolConfig:
    market: str = "crypto"
    timeframe: str = "1h"
    candles: int = 200
    paper_mode: bool = True
    watchlist: Tuple[str, ...] = ("BTC", "ETH", "SOL")
    volume: VolumeTierConfig = VolumeTierConfig()
    whale: WhaleFlagConfig = WhaleFlagConfig()


# ======================================
# DATA PROVIDER
# ======================================

class MarketProvider:
    def __init__(self):
        self.exchange = ccxt.okx()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        pair = f"{symbol}/USDT"
        return self.exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)


# ======================================
# INDICATORS
# ======================================

def relative_volume(df: pd.DataFrame) -> float:
    vol = df["volume"]
    if len(vol) < 20:
        return 0.0
    return float(vol.iloc[-1] / vol.iloc[-20:].mean())


def atr_pct_from_df(df: pd.DataFrame) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    return float(atr.iloc[-1] / close.iloc[-1])


def trend_state(df: pd.DataFrame) -> str:
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()

    if fast.iloc[-1] > slow.iloc[-1]:
        return "Bullish"
    if fast.iloc[-1] < slow.iloc[-1]:
        return "Bearish"
    return "Neutral"


def whale_flags(df: pd.DataFrame, cfg: WhaleFlagConfig) -> Dict[str, bool]:
    return {
        "absorption": False,
        "stealth_accumulation": False,
        "wick_defense": False
    }


def detect_quiet_accumulation(
    rvol: float,
    atr_pct: float,
    trend: str,
    whale_count: int,
    cfg: VolumeTierConfig
) -> bool:
    if whale_count >= 2:
        return False
    if rvol >= cfg.tier_b_min_rvol:
        return False
    if atr_pct > 0.008:
        return False
    if trend not in ("Bullish", "Bearish", "Neutral"):
        return False
    return True


# ======================================
# TIER LOGIC
# ======================================

def classify_tier(trend: str, rvol: float, whale_count: int, cfg: VolumeTierConfig) -> str:
    if trend not in ("Bullish", "Bearish"):
        return "C"

    if rvol >= cfg.tier_a_rvol:
        return "A"

    if whale_count >= 2 and rvol >= cfg.tier_bplus_min_rvol:
        return "B+"

    if whale_count == 1 and rvol >= cfg.tier_b_min_rvol:
        return "B+"

    if rvol >= cfg.tier_b_min_rvol:
        return "B"

    return "C"


# ======================================
# EVALUATION
# ======================================

def evaluate_symbol(symbol: str, cfg: ToolConfig) -> Dict:
    provider = MarketProvider()
    raw = provider.fetch_ohlcv(symbol, cfg.timeframe, cfg.candles)

    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])

    rvol = relative_volume(df)
    atr_pct = atr_pct_from_df(df)
    trend = trend_state(df)

    flags = whale_flags(df, cfg.whale)
    whale_count = sum(flags.values())

    tier = classify_tier(trend, rvol, whale_count, cfg.volume)

    quiet = detect_quiet_accumulation(
        rvol=rvol,
        atr_pct=atr_pct,
        trend=trend,
        whale_count=whale_count,
        cfg=cfg.volume
    )

    return {
        "symbol": symbol,
        "trend": trend,
        "tier": tier,
        "rvol": round(rvol, 2),
        "atr_pct": round(atr_pct, 4),
        "whale_flags": flags,
        "whale_count": whale_count,
        "quiet_accumulation": quiet,
        "decision": "ALERT" if tier in ("A", "B+") else "LOG",
        "paper_mode": cfg.paper_mode
    }


# ======================================
# FLASK APP
# ======================================

def build_flask_app(cfg: ToolConfig) -> Flask:
    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/scout")
    def scout():
        results = []
        for sym in cfg.watchlist:
            try:
                results.append(evaluate_symbol(sym, cfg))
            except Exception as e:
                results.append({"symbol": sym, "error": str(e)})
        return jsonify(results)

    @app.route("/eval")
    def eval_manual():
        symbol = request.args.get("symbol", "").upper()
        if not symbol:
            return jsonify({"error": "Usage: /eval?symbol=BTC"}), 400
        return jsonify(evaluate_symbol(symbol, cfg))

    return app


# ======================================
# ENTRYPOINT
# ======================================

if __name__ == "__main__":
    cfg = ToolConfig()

    run_mode = os.environ.get("RUN_MODE", "server")

    if run_mode == "server":
        app = build_flask_app(cfg)
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port)



        



   


  

