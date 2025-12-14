trade_evaluator_complete.py
# trade_evaluator_complete.py
# FULL Trade Evaluator Tool
# Tiered Volume + Whale Detection + Risk + Auto Scout
# Paper Trading SAFE

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# =========================
# OPTIONAL SCHEDULER
# =========================
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:
    BlockingScheduler = None
    CronTrigger = None

# =========================
# TIMEZONE
# =========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

CHI_TZ = "America/Chicago"


# =========================
# CONFIG STRUCTS
# =========================

@dataclass
class VolumeThresholds:
    tier_a_rvol: float = 1.50     # breakout / public
    tier_bplus_min: float = 1.05  # whale accumulation
    tier_b_min: float = 0.80      # early allowed


@dataclass
class WhaleParams:
    vol_rise_min: float = 1.10
    atr_fall_pct: float = 0.03
    absorption_vol_min: float = 1.00


@dataclass
class TradeConfig:
    bankroll: float = 3000.0
    risk_per_trade: float = 0.03
    min_rr: float = 2.0
    paper_mode: bool = True
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
# CORE ENGINE
# =========================

class TradeEvaluator:

    def __init__(self, cfg: TradeConfig):
        self.cfg = cfg

    # ----- Indicators -----

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def relative_volume(df: pd.DataFrame, period: int = 20) -> float:
        return float(df["volume"].iloc[-1] / df["volume"].rolling(period).mean().iloc[-1])

    # ----- Trend -----

    def trend_state(self, df: pd.DataFrame) -> str:
        sma30 = self.sma(df["close"], 30)
        sma50 = self.sma(df["close"], 50)
        if sma30.iloc[-1] > sma50.iloc[-1]:
            return "Bullish"
        if sma30.iloc[-1] < sma50.iloc[-1]:
            return "Bearish"
        return "Neutral"

    # ----- Whale Detection -----

    def whale_flags(self, df: pd.DataFrame) -> Dict[str, bool]:
        rvol = self.relative_volume(df)
        atr_series = self.atr(df)
        atr_now = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-6]

        stealth = rvol >= self.cfg.whale.vol_rise_min and atr_now < atr_prev
        absorption = rvol >= self.cfg.whale.absorption_vol_min

        return {
            "stealth_accumulation": stealth,
            "absorption": absorption
        }

    # ----- Tier Logic -----

    def classify_tier(self, trend: str, rvol: float, whale_count: int) -> str:
        v = self.cfg.volume
        if trend not in ("Bullish", "Bearish"):
            return "C"
        if rvol >= v.tier_a_rvol:
            return "A"
        if rvol >= v.tier_bplus_min and whale_count >= 1:
            return "B+"
        if rvol >= v.tier_b_min:
            return "B"
        return "C"

    # ----- Risk -----

    def position_size(self, entry: float, stop: float) -> float:
        risk_amt = self.cfg.bankroll * self.cfg.risk_per_trade
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            return 0.0
        return risk_amt / risk_per_unit

    @staticmethod
    def rr_ratio(entry: float, stop: float, target: float) -> float:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0:
            return 0.0
        return round(reward / risk, 2)

    # ----- Evaluation -----

    def evaluate_trade(
        self,
        df: pd.DataFrame,
        entry: float,
        stop: float,
        target: float,
        symbol: str
    ) -> Dict[str, object]:

        trend = self.trend_state(df)
        rvol = self.relative_volume(df)
        flags = self.whale_flags(df)
        whale_count = sum(flags.values())
        tier = self.classify_tier(trend, rvol, whale_count)

        rr = self.rr_ratio(entry, stop, target)
        size = self.position_size(entry, stop)

        decision = "WATCH"
        if tier == "A" and rr >= self.cfg.min_rr:
            decision = "HIGH QUALITY"
        elif tier in ("B+", "B") and rr >= self.cfg.min_rr:
            decision = "EARLY / WHALE"
        elif tier == "C":
            decision = "IGNORE"

        return {
            "symbol": symbol,
            "time": now_chicago().isoformat(timespec="seconds"),
            "trend": trend,
            "tier": tier,
            "rvol": round(rvol, 2),
            "whale_flags": flags,
            "rr": rr,
            "position_size": round(size, 2),
            "decision": decision,
            "paper_mode": self.cfg.paper_mode
        }


# =========================
# DATA SOURCE (PLUG IN)
# =========================

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    MUST return DataFrame with columns:
    open, high, low, close, volume
    """
    raise NotImplementedError("Connect your data source here")


# =========================
# AUTO SCOUT
# =========================

def auto_scout(
    evaluator: TradeEvaluator,
    watchlist: List[str],
    entries: Optional[Dict[str, Tuple[float, float, float]]] = None
) -> None:

    for symbol in watchlist:
        df = fetch_ohlcv(symbol)

        if entries and symbol in entries:
            entry, stop, target = entries[symbol]
            result = evaluator.evaluate_trade(df, entry, stop, target, symbol)
            print(result)
        else:
            trend = evaluator.trend_state(df)
            rvol = evaluator.relative_volume(df)
            flags = evaluator.whale_flags(df)
            tier = evaluator.classify_tier(trend, rvol, sum(flags.values()))

            print(
                f"{symbol:>6} | "
                f"Tier {tier} | "
                f"Trend {trend:<7} | "
                f"RVOL {round(rvol,2)} | "
                f"Whales {sum(flags.values())}"
            )


# =========================
# SCHEDULER
# =========================

def run_scheduler(evaluator: TradeEvaluator, watchlist: List[str]) -> None:
    if BlockingScheduler is None:
        raise RuntimeError("Install apscheduler")

    scheduler = BlockingScheduler(timezone=CHI_TZ)
    trigger = CronTrigger(day_of_week="mon-fri", hour="7-20", minute=0)

    def job():
        print("\nAUTO SCOUT —", now_chicago().strftime("%Y-%m-%d %H:%M:%S"), "CST")
        print("-" * 50)
        auto_scout(evaluator, watchlist)

    scheduler.add_job(job, trigger)
    scheduler.start()


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    config = TradeConfig(paper_mode=True)
    evaluator = TradeEvaluator(config)

    WATCHLIST = ["AAPL", "NVDA", "AMD"]

    print("FULL Trade Evaluator loaded (paper trading ON)")
    # run_scheduler(evaluator, WATCHLIST)

   


  

