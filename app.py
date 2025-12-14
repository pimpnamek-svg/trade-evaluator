"""trade_evaluator_tool.py

Trade Evaluator Tool (Stocks/Crypto) — v2 (Tiered Volume + Whale Flags + Auto Scout)

What changed vs v1
- Volume is no longer a hard veto. It classifies setups into tiers (A / B+ / B / C).
- Added "whale activity" flags (stealth accumulation / absorption style behavior).
- Added auto-scout mode: runs every hour, Mon–Fri, 7:00–20:00 America/Chicago.
- Added paper trading safety: paper_mode=True prevents any execution (log/alerts only).

Notes
- This file is evaluator + scheduler ready. You still need a real data source.
  By default, it expects a pandas DataFrame with columns: open, high, low, close, volume.
- You can wire a data provider (yfinance, polygon, alpaca, schwab API, etc.) in `fetch_ohlcv()`.

"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependency for scheduling (recommended)
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:
    BlockingScheduler = None
    CronTrigger = None

# Timezone handling
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


# ----------------------------
# Config
# ----------------------------
CHI_TZ = "America/Chicago"


@dataclass
class VolumeThresholds:
    """Tune these based on your market / timeframe."""

    tier_a_rvol: float = 1.50   # confirmed momentum
    tier_bplus_min: float = 1.05  # quiet accumulation baseline
    tier_b_min: float = 0.80    # early/signal-style allowed


@dataclass
class WhaleParams:
    """Lightweight whale flag heuristics (keep simple while paper testing)."""

    # Flag 1: volume rising while ATR falls (stealth accumulation)
    vol_rise_min: float = 1.10
    atr_fall_pct: float = 0.03  # ATR fell at least 3% vs prior window

    # Flag 2: absorption (price compressed + >= baseline vol)
    compression_pct: float = 0.012  # last-N range / close <= 1.2%
    absorption_vol_min: float = 1.00

    # Flag 3: wick defense (long lower wicks + vol not collapsing)
    wick_ratio_min: float = 1.50  # lower wick >= 1.5x body
    wick_checks: int = 5
    wick_vol_floor: float = 0.90


@dataclass
class TradeEvaluatorConfig:
    bankroll: float = 3000.0
    risk_per_trade: float = 0.03
    paper_mode: bool = True

    # scoring weights
    w_trend: int = 30
    w_rr: int = 25
    w_volume_confirm: int = 25
    w_whale_bonus: int = 10

    # thresholds
    min_rr: float = 2.0
    volume: VolumeThresholds = VolumeThresholds()
    whale: WhaleParams = WhaleParams()


# ----------------------------
# Utilities
# ----------------------------

def now_chicago() -> datetime:
    if ZoneInfo is None:
        # fallback: local time
        return datetime.now()
    return datetime.now(ZoneInfo(CHI_TZ))


def play_cash_register_sound() -> None:
    """Plays a simple cash register alert.

    Cross-platform approach:
    - Windows: winsound (built-in)
    - macOS/Linux: terminal bell fallback

    You can replace this with an actual .wav file player later.
    """
    try:
        if sys.platform.startswith("win"):
            import winsound  # type: ignore
            # Frequency, duration (ms) — a quick "cha-ching" style beep pattern
            winsound.Beep(880, 120)
            winsound.Beep(1320, 120)
            winsound.Beep(880, 180)
        else:
            # Terminal bell
            print("", end="", flush=True)
    except Exception:
        pass


# ----------------------------
# Core Evaluator
# ----------------------------


class TradeEvaluator:
    def __init__(self, cfg: TradeEvaluatorConfig):
        self.cfg = cfg

    # ---------- Indicators ----------
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
        avg_vol = df["volume"].rolling(period).mean()
        return df["volume"] / avg_vol

    # ---------- Market State ----------
    def trend_state(self, df: pd.DataFrame) -> str:
        sma30 = self.sma(df["close"], 30)
        sma50 = self.sma(df["close"], 50)
        if np.isnan(sma30.iloc[-1]) or np.isnan(sma50.iloc[-1]):
            return "Unknown"
        if sma30.iloc[-1] > sma50.iloc[-1]:
            return "Bullish"
        if sma30.iloc[-1] < sma50.iloc[-1]:
            return "Bearish"
        return "Neutral"

    def volatility_state(self, df: pd.DataFrame) -> Tuple[str, float]:
        atr_val = self.atr(df).iloc[-1]
        close = df["close"].iloc[-1]
        if close == 0 or np.isnan(atr_val):
            return "Unknown", float("nan")
        atr_pct = atr_val / close
        if atr_pct > 0.05:
            return "High Volatility", float(atr_pct)
        if atr_pct > 0.02:
            return "Moderate Volatility", float(atr_pct)
        return "Low Volatility", float(atr_pct)

    # ---------- Risk & Position Sizing ----------
    def position_size(self, entry: float, stop: float) -> float:
        risk_amount = self.cfg.bankroll * self.cfg.risk_per_trade
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0
        return risk_amount / stop_distance

    @staticmethod
    def rr_ratio(entry: float, stop: float, target: float) -> float:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    # ---------- Whale Flags ----------
    def whale_flags(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Returns a dict of whale flags.

        Designed to be lightweight and robust for paper testing.
        """
        wp = self.cfg.whale

        rvol_series = self.relative_volume(df)
        rvol = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else 0.0

        atr_series = self.atr(df)
        atr_now = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
        atr_prev = float(atr_series.iloc[-6]) if len(atr_series) > 6 and not np.isnan(atr_series.iloc[-6]) else atr_now

        # Flag 1: volume rising while ATR falls (stealth accumulation)
        atr_fell = atr_now <= (atr_prev * (1.0 - wp.atr_fall_pct)) if atr_prev > 0 else False
        flag_stealth = (rvol >= wp.vol_rise_min) and atr_fell

        # Flag 2: absorption — tight range with baseline+ volume
        n = 10
        if len(df) >= n:
            recent = df.iloc[-n:]
            rng = float(recent["high"].max() - recent["low"].min())
            close = float(df["close"].iloc[-1]) if df["close"].iloc[-1] != 0 else 1.0
            compression = (rng / close) <= wp.compression_pct
        else:
            compression = False
        flag_absorption = compression and (rvol >= wp.absorption_vol_min)

        # Flag 3: wick defense — repeated long lower wicks + volume not collapsing
        m = wp.wick_checks
        wick_ok = False
        if len(df) >= m:
            recent = df.iloc[-m:]
            bodies = (recent["close"] - recent["open"]).abs()
            lower_wicks = (recent[["open", "close"]].min(axis=1) - recent["low"]).clip(lower=0)
            # avoid divide by zero
            ratio = lower_wicks / (bodies.replace(0, np.nan))
            wick_count = int((ratio >= wp.wick_ratio_min).sum())
            wick_ok = (wick_count >= max(2, m // 2)) and (rvol >= wp.wick_vol_floor)
        flag_wick_defense = wick_ok

        return {
            "stealth_accumulation": bool(flag_stealth),
            "absorption": bool(flag_absorption),
            "wick_defense": bool(flag_wick_defense),
        }

    # ---------- Tiering ----------
    def classify_tier(self, trend: str, rvol: float, whale_flag_count: int) -> str:
        vt = self.cfg.volume

        # You can still tier neutrals, but usually you want trend alignment.
        if trend not in ("Bullish", "Bearish"):
            return "C"

        if rvol >= vt.tier_a_rvol:
            return "A"

        # Whale-aware early tier
        if (rvol >= vt.tier_bplus_min) and (whale_flag_count >= 2):
            return "B+"

        if rvol >= vt.tier_b_min:
            return "B"

        return "C"

    # ---------- Evaluation ----------
    def evaluate_trade(
        self,
        df: pd.DataFrame,
        entry: float,
        stop: float,
        target: float,
        symbol: str = "",
    ) -> Dict[str, object]:
        """Evaluate a single setup (manual mode)."""

        # Basic validation
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

        trend = self.trend_state(df)
        vol_state, atr_pct = self.volatility_state(df)

        rvol_series = self.relative_volume(df)
        rvol = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else 0.0

        flags = self.whale_flags(df)
        whale_count = int(sum(flags.values()))

        tier = self.classify_tier(trend=trend, rvol=rvol, whale_flag_count=whale_count)

        rr = self.rr_ratio(entry, stop, target)
        size = self.position_size(entry, stop)

        # Scoring (direction + RR + confirmation + whale bonus)
        score = 0
        if trend == "Bullish":
            score += self.cfg.w_trend
        elif trend == "Bearish":
            score += int(self.cfg.w_trend * 0.66)  # slightly lower unless you want symmetrical

        if rr >= self.cfg.min_rr:
            score += self.cfg.w_rr

        if rvol >= self.cfg.volume.tier_a_rvol:
            score += self.cfg.w_volume_confirm

        if whale_count >= 2:
            score += self.cfg.w_whale_bonus

        # Decision logic: never hard-skip just for volume; tier + score guides behavior
        if tier == "A" and score >= 70:
            decision = "High-Quality (Confirmed)"
        elif tier in ("B+", "B") and score >= 50:
            decision = "Early / Signal-Style (Paper-Test Friendly)"
        elif tier == "C":
            decision = "Watch Only"
        else:
            decision = "Skip / Low Edge"

        return {
            "Symbol": symbol,
            "Timestamp (Chicago)": now_chicago().isoformat(timespec="seconds"),
            "Trend": trend,
            "Volatility": vol_state,
            "ATR%": (round(atr_pct, 4) if not np.isnan(atr_pct) else None),
            "Rel Volume": round(rvol, 2),
            "Tier": tier,
            "Whale Flags": flags,
            "Whale Flag Count": whale_count,
            "RR": rr,
            "Position Size (shares/units)": round(float(size), 4),
            "Score": int(score),
            "Decision": decision,
            "Paper Mode": bool(self.cfg.paper_mode),
        }


# ----------------------------
# Auto-Scout (Hourly)
# ----------------------------

def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    """DATA SOURCE PLACEHOLDER.

    Replace this with your real data provider.

    Must return a DataFrame with columns: open, high, low, close, volume
    indexed by time (recommended) or with a time column.

    Examples you can wire later:
    - yfinance (stocks)
    - polygon, alpaca, schwab API
    - ccxt (crypto)
    """
    raise NotImplementedError("Wire your data source in fetch_ohlcv().")


def auto_scout(
    evaluator: TradeEvaluator,
    watchlist: List[str],
    entry_stop_target: Optional[Dict[str, Tuple[float, float, float]]] = None,
    alert_on_tiers: Tuple[str, ...] = ("A", "B+"),
) -> List[Dict[str, object]]:
    """Scans the watchlist and returns ranked results.

    entry_stop_target: optional dict symbol -> (entry, stop, target)
      If not provided, the scout can still tier/flag whales, but RR/size will be skipped.

    Returns: list of result dicts.
    """

    results: List[Dict[str, object]] = []

    for sym in watchlist:
        try:
            df = fetch_ohlcv(sym)
        except NotImplementedError:
            # If data source isn't wired, stop cleanly so you notice immediately.
            raise
        except Exception as e:
            results.append({
                "Symbol": sym,
                "Error": str(e),
            })
            continue

        if entry_stop_target and sym in entry_stop_target:
            entry, stop, target = entry_stop_target[sym]
            res = evaluator.evaluate_trade(df, entry, stop, target, symbol=sym)
        else:
            # Tier/whale-only mode (no entry/stop/target yet)
            trend = evaluator.trend_state(df)
            vol_state, atr_pct = evaluator.volatility_state(df)
            rvol_series = evaluator.relative_volume(df)
            rvol = float(rvol_series.iloc[-1]) if not np.isnan(rvol_series.iloc[-1]) else 0.0
            flags = evaluator.whale_flags(df)
            whale_count = int(sum(flags.values()))
            tier = evaluator.classify_tier(trend=trend, rvol=rvol, whale_flag_count=whale_count)
            res = {
                "Symbol": sym,
                "Timestamp (Chicago)": now_chicago().isoformat(timespec="seconds"),
                "Trend": trend,
                "Volatility": vol_state,
                "ATR%": (round(atr_pct, 4) if not np.isnan(atr_pct) else None),
                "Rel Volume": round(rvol, 2),
                "Tier": tier,
                "Whale Flags": flags,
                "Whale Flag Count": whale_count,
                "Decision": "Alert" if tier in alert_on_tiers else "Log",
                "Paper Mode": bool(evaluator.cfg.paper_mode),
            }

        results.append(res)

    # Rank results: Tier (A > B+ > B > C) then whale count then rvol
    tier_rank = {"A": 3, "B+": 2, "B": 1, "C": 0}

    def key_fn(r: Dict[str, object]):
        t = str(r.get("Tier", "C"))
        return (
            tier_rank.get(t, 0),
            int(r.get("Whale Flag Count", 0) or 0),
            float(r.get("Rel Volume", 0) or 0),
        )

    results.sort(key=key_fn, reverse=True)

    # Alerts
    top_alerts = [r for r in results if str(r.get("Tier")) in alert_on_tiers]
    if top_alerts:
        play_cash_register_sound()

    return results


# ----------------------------
# Scheduler (Mon–Fri hourly, 7a–8p CST)
# ----------------------------

def run_scheduler(evaluator: TradeEvaluator, watchlist: List[str]) -> None:
    if BlockingScheduler is None or CronTrigger is None:
        raise RuntimeError(
            "APScheduler is not installed. Install it with: pip install apscheduler"
        )

    scheduler = BlockingScheduler(timezone=CHI_TZ)

    # Run at minute 0 of every hour 7–20 (7am to 8pm), Monday–Friday
    trigger = CronTrigger(day_of_week="mon-fri", hour="7-20", minute=0)

    def job():
        stamp = now_chicago().strftime("%Y-%m-%d %H:%M:%S")
        print(f"
AUTO SCOUT — {stamp} CST")
        print("-" * 45)
        try:
            results = auto_scout(evaluator, watchlist)
        except NotImplementedError:
            print("fetch_ohlcv() is not wired yet. Add your data provider and restart.")
            return

        for r in results[:10]:
            sym = r.get("Symbol")
            tier = r.get("Tier")
            trend = r.get("Trend")
            rvol = r.get("Rel Volume")
            whales = r.get("Whale Flag Count")
            decision = r.get("Decision")
            print(f"{sym:>8} | Tier {tier:<2} | Trend {trend:<7} | RVOL {rvol:<5} | Whale {whales} | {decision}")

    scheduler.add_job(job, trigger=trigger, id="auto_scout_hourly")

    print("Scheduler started: hourly Mon–Fri 7:00–20:00 America/Chicago")
    print("Press Ctrl+C to stop.")
    scheduler.start()


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # Paper trading safety ON by default
    cfg = TradeEvaluatorConfig(bankroll=3000.0, risk_per_trade=0.03, paper_mode=True)
    evaluator = TradeEvaluator(cfg)

    # 1) Manual evaluation example (requires your DataFrame)
    # df = fetch_ohlcv("AAPL")
    # print(evaluator.evaluate_trade(df, entry=190.0, stop=185.0, target=205.0, symbol="AAPL"))

    # 2) Auto scout example
    watchlist = [
        # Add your symbols here
        "AAPL", "NVDA", "AMD"
    ]

    # Choose what to run:
    # - run_scheduler() for hourly auto scout
    # - auto_scout() for one-off scan

    # One-off scan (will error until fetch_ohlcv is implemented)
    # print(auto_scout(evaluator, watchlist)[:5])

    # Hourly schedule (will error until fetch_ohlcv is implemented)
    # run_scheduler(evaluator, watchlist)

    print("Loaded trade_evaluator_tool.py (v2). Wire fetch_ohlcv() to your data source, then run.")

