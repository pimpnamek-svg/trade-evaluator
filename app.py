# Trade Evaluator Tool — v2.1 (Tiered Volume + Whale Flags + Auto Scout)

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# Optional scheduler
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:
    BlockingScheduler = None
    CronTrigger = None

# Timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

CHI_TZ = "America/Chicago"


# =========================
# CONFIG STRUCTURES
# =========================

@dataclass
class VolumeThresholds:
    tier_a_rvol: float = 1.50     # breakout / public momentum
    tier_bplus_min: float = 1.05  # quiet accumulation
    tier_b_min: float = 0.80      # early signal allowed


@dataclass
class WhaleParams:
    vol_rise_min: float = 1.10
    atr_fall_pct: float = 0.03
    absorption_vol_min: float = 1.00


@dataclass
class TradeEvaluatorConfig:
    bankroll: float = 3000.0
    risk_per_trade: float = 0.03
    paper_mode: bool = True
    min_rr: float = 2.0
    volume: VolumeThresholds = VolumeThresholds()
    whale: WhaleParams = WhaleParams()


# =========================
# UTILITIES
# =========================

def now_chicago() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    return datetime.now(ZoneInfo(CHI_TZ))


# =========================
# CORE EVALUATOR
# =========================

class TradeEvaluator:

    def __init__(self, cfg: TradeEvaluatorConfig):
        self.cfg = cfg

    # ----- Indicators -----

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["volume"] / df["volume"].rolling(period).mean()

    # ----- Market State -----

    def trend_state(self, df: pd.DataFrame) -> str:
        sma30 = self.sma(df["close"], 30)
        sma50 = self.sma(df["close"], 50)

        if sma30.iloc[-1] > sma50.iloc[-1]:
            return "Bullish"
        elif sma30.iloc[-1] < sma50.iloc[-1]:
            return "Bearish"
        return "Neutral"

    # ----- Whale Detection -----

    def whale_flags(self, df: pd.DataFrame) -> Dict[str, bool]:
        rvol = self.relative_volume(df).iloc[-1]
        atr_series = self.atr(df)

        atr_now = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-6]

        stealth = rvol >= self.cfg.whale.vol_rise_min and atr_now < atr_prev
        absorption = rvol >= self.cfg.whale.absorption_vol_min

        return {
            "stealth_accumulation": bool(stealth),
            "absorption": bool(absorption)
        }

    # ----- Tier Logic -----

    def classify_tier(self, trend: str, rvol: float, whale_count: int) -> str:
        vt = self.cfg.volume

        if trend not in ("Bullish", "Bearish"):
            return "C"

        if rvol >= vt.tier_a_rvol:
            return "A"

        if rvol >= vt.tier_bplus_min and whale_count >= 1:
            return "B+"

        if rvol >= vt.tier_b_min:
            return "B"

        return "C"


# =========================
# DATA SOURCE PLACEHOLDER
# =========================

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Replace this with your real data source.
    Must return DataFrame with:
    open, high, low, close, volume
    """
    raise NotImplementedError("Connect your data source here")


# =========================
# AUTO SCOUT
# =========================

def auto_scout(evaluator: TradeEvaluator, watchlist: List[str]) -> None:
    for symbol in watchlist:
        df = fetch_ohlcv(symbol)

        trend = evaluator.trend_state(df)
        rvol = evaluator.relative_volume(df).iloc[-1]
        flags = evaluator.whale_flags(df)
        whale_count = sum(flags.values())

        tier = evaluator.classify_tier(trend, rvol, whale_count)

        print(
            f"{symbol:>6} | "
            f"Tier {tier} | "
            f"Trend {trend:<7} | "
            f"RVOL {round(rvol,2)} | "
            f"Whales {whale_count}"
        )


# =========================
# SCHEDULER
# =========================

def run_scheduler(evaluator: TradeEvaluator, watchlist: List[str]) -> None:
    if BlockingScheduler is None:
        raise RuntimeError("Install apscheduler first")

    scheduler = BlockingScheduler(timezone=CHI_TZ)
    trigger = CronTrigger(day_of_week="mon-fri", hour="7-20", minute=0)

    def job():
        print("\nAUTO SCOUT —", now_chicago().strftime("%Y-%m-%d %H:%M:%S"), "CST")
        print("-" * 45)
        auto_scout(evaluator, watchlist)

    scheduler.add_job(job, trigger)
    scheduler.start()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    cfg = TradeEvaluatorConfig(paper_mode=True)
    evaluator = TradeEvaluator(cfg)

    watchlist = ["AAPL", "NVDA", "AMD"]

    print("Trade Evaluator v2.1 loaded successfully (paper mode ON)")
    # run_scheduler(evaluator, watchlist)


