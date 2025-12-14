"""
# force rebuild

TRADE EVALUATOR TOOL — Reconstructed "Original-Style" Monolithic Script (v0.REBUILD)

Goal:
- Reconstruct a full-sized, all-in-one evaluator tool (similar to your original 455+ lines).
- Add your requested enhancements WITHOUT shrinking the tool down into a tiny demo.

What you asked to add:
1) Volume requirements less strict (manual + automatic)
2) Tiered setups (A / B+ / B / C)
3) Whale activity flagging (stealth accumulation / absorption / wick defense)
4) Auto-picker: scout every hour Monday–Friday 7:00a–8:00p CST
5) Alert sound: cash register style (beep fallback is used here)
6) Paper trading caveat (NO execution)

This script includes:
- Indicator engine (SMA, ATR, RVOL)
- Structure scoring & decision logic
- Tier engine (volume as classifier, NOT veto)
- Whale flag engine
- Risk manager (position size, ATR stops, RR)
- Watchlist auto-scanner (hourly schedule in CST without extra libs)
- Logging to CSV (paper log)
- Data sources:
  - Stocks: Stooq CSV (requests) — free, no API key
  - Crypto: ccxt — requires exchange connectivity; uses env CRYPTO_EXCHANGE (default binance)

Dependencies (requirements.txt):
- pandas
- numpy
- requests
- ccxt     (only if you use crypto)
- flask    (only if you want server mode; optional)

IMPORTANT:
- This is a "hard reset" rebuild. It should run cleanly.
- If your environment is Railway, make sure you run it as a web server OR run it as a worker.
  CLI won't show in a web URL. See "RAILWAY NOTES" near the bottom.

Author: ChatGPT (rebuild)
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import csv
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

# Optional crypto dependency
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # allows stocks-only usage

# Optional server dependency (Railway-friendly mode)
try:
    from flask import Flask, jsonify, request as flask_request  # type: ignore
except Exception:
    Flask = None
    jsonify = None
    flask_request = None


# ============================================================
# CONFIG & CONSTANTS
# ============================================================

TZ_NAME = "America/Chicago"  # CST/CDT handling via zoneinfo when available

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


MarketType = Literal["stock", "crypto"]
TierType = Literal["A", "B+", "B", "C"]


@dataclass
class VolumeTierConfig:
    """
    Tier logic:
      A: Confirmed momentum (public breakout) -> high RVOL
      B+: Whale/accumulation style -> modest RVOL + whale flags
      B: Early/signal style -> low/moderate RVOL allowed
      C: Watch only / low edge
    """
    tier_a_rvol: float = 1.50
    tier_bplus_min_rvol: float = 1.05
    tier_b_min_rvol: float = 0.80

    # If you want to allow even earlier entries:
    # tier_b_min_rvol could be 0.65, but start with 0.80 while testing.


@dataclass
class WhaleFlagConfig:
    """
    Whale flag heuristics (lightweight, paper-test friendly).
    These are probabilistic signals, not guarantees.
    """
    # Flag 1: stealth accumulation
    stealth_rvol_min: float = 1.10
    atr_fall_lookback: int = 6
    atr_fall_pct: float = 0.03  # ATR fell >= 3%

    # Flag 2: absorption (tight range + baseline volume)
    absorption_rvol_min: float = 1.00
    compression_window: int = 10
    compression_pct: float = 0.012  # range/close <= 1.2%

    # Flag 3: wick defense
    wick_checks: int = 6
    lower_wick_ratio_min: float = 1.50
    wick_vol_floor: float = 0.90  # don't want volume collapsing


@dataclass
class RiskConfig:
    bankroll: float = 3000.0
    risk_per_trade: float = 0.03  # 3% default (you can change)
    min_rr: float = 2.0
    use_atr_stop: bool = True
    atr_period: int = 14
    atr_stop_mult: float = 1.5  # typical 1.0–2.5 depending on style
    round_shares: bool = True


@dataclass
class ScoutScheduleConfig:
    enabled: bool = True
    # Monday-Friday only:
    weekdays_only: bool = True
    # 7a CST to 8p CST inclusive hours:
    start_hour: int = 7
    end_hour: int = 20
    # run on minute 0 of each hour (top of hour)
    minute: int = 0
    # interval guard (if system sleeps/late)
    min_seconds_between_runs: int = 55 * 60  # ~55 min


@dataclass
class ToolConfig:
    market: MarketType = "stock"
    timeframe: str = "1h"
    candles: int = 220
    paper_mode: bool = True
    play_alert_sound: bool = True

    volume: VolumeTierConfig = field(default_factory=VolumeTierConfig)
    whale: WhaleFlagConfig = field(default_factory=WhaleFlagConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    schedule: ScoutScheduleConfig = field(default_factory=ScoutScheduleConfig)

    log_dir: str = "./logs"
    scout_log_csv: str = "scout_results.csv"
    eval_log_csv: str = "manual_evals.csv"

    watchlist: tuple[str, ...] = (
        "AAPL", "NVDA", "AMD", "TSLA", "MSFT", "AMZN"
    )



# ============================================================
# TIME / TIMEZONE HELPERS
# ============================================================

def now_cst() -> datetime:
    """
    Returns current time in America/Chicago if zoneinfo available,
    else returns local time (still workable for scheduling if server is CST).
    """
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(TZ_NAME))
    except Exception:
        return datetime.now()


def is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5  # Mon=0 .. Fri=4


def within_schedule_window(dt: datetime, sched: ScoutScheduleConfig) -> bool:
    if sched.weekdays_only and not is_weekday(dt):
        return False
    return sched.start_hour <= dt.hour <= sched.end_hour


def is_top_of_hour(dt: datetime, sched: ScoutScheduleConfig) -> bool:
    return dt.minute == sched.minute


# ============================================================
# SOUND ALERT (cash register-ish fallback)
# ============================================================

def cash_register_alert():
    """
    Simple alert:
    - Windows: winsound beep pattern
    - Others: terminal bell
    """
    try:
        if sys.platform.startswith("win"):
            import winsound  # type: ignore
            winsound.Beep(880, 120)
            winsound.Beep(1320, 120)
            winsound.Beep(880, 200)
        else:
            # Terminal bell
            print("\a", end="", flush=True)
    except Exception:
        pass


# ============================================================
# DATA PROVIDERS
# ============================================================

class DataProviderError(Exception):
    pass


class BaseProvider:
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError


class StooqStockProvider(BaseProvider):
    """
    Stock OHLCV from Stooq (free, no API key).
    Stooq provides daily and some intraday; intraday coverage can be limited.
    If intraday isn't available for a symbol, you may get empty/limited data.
    """
    BASE = "https://stooq.com/q/d/l/"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Stooq intraday is not consistent; we treat timeframe as best-effort.
        # We'll fetch daily CSV and let you still evaluate swing signals.
        params = {"s": symbol.lower(), "i": "d"}  # daily
        try:
            r = requests.get(self.BASE, params=params, timeout=20)
            r.raise_for_status()
        except Exception as e:
            raise DataProviderError(f"Stooq request failed: {e}") from e

        # Parse CSV
        from io import StringIO
        raw = StringIO(r.text)
        df = pd.read_csv(raw)

        if df.empty or "Close" not in df.columns:
            raise DataProviderError(f"No data returned for {symbol} from Stooq.")

        # Normalize columns
        df.rename(
            columns={
                "Date": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        df = df.tail(limit).reset_index(drop=True)

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        if len(df) < 60:
            raise DataProviderError(f"Not enough bars for {symbol}: got {len(df)}")

        return df


class CCXTCryptoProvider(BaseProvider):
    """
    Crypto OHLCV from ccxt.
    Requires:
      - ccxt installed
      - network access
    Uses env var CRYPTO_EXCHANGE (default 'binance').
    """

    def __init__(self):
        if ccxt is None:
            raise DataProviderError("ccxt is not installed. Add it to requirements and redeploy.")
        ex_name = os.environ.get("CRYPTO_EXCHANGE", "binance").lower()
        if not hasattr(ccxt, ex_name):
            raise DataProviderError(f"Unknown ccxt exchange: {ex_name}")
        self.exchange = getattr(ccxt, ex_name)({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Ensure symbol format like "BTC/USDT"
        if "/" not in symbol:
            symbol = f"{symbol.upper()}/USDT"
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            raise DataProviderError(f"ccxt fetch_ohlcv failed for {symbol}: {e}") from e

        if not ohlcv:
            raise DataProviderError(f"No OHLCV returned for {symbol}.")

        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

        if len(df) < 60:
            raise DataProviderError(f"Not enough bars for {symbol}: got {len(df)}")

        return df


def build_provider(market: MarketType) -> BaseProvider:
    if market == "stock":
        return StooqStockProvider()
    if market == "crypto":
        return CCXTCryptoProvider()
    raise ValueError(f"Unsupported market: {market}")


# ============================================================
# INDICATORS & MARKET STATE
# ============================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def relative_volume(df: pd.DataFrame, period: int = 20) -> float:
    avg_vol = df["volume"].rolling(period).mean().iloc[-1]
    if avg_vol == 0 or pd.isna(avg_vol):
        return 0.0
    return float(df["volume"].iloc[-1] / avg_vol)


def trend_state(df: pd.DataFrame) -> str:
    s30 = sma(df["close"], 30).iloc[-1]
    s50 = sma(df["close"], 50).iloc[-1]
    if pd.isna(s30) or pd.isna(s50):
        return "Unknown"
    if s30 > s50:
        return "Bullish"
    if s30 < s50:
        return "Bearish"
    return "Neutral"


def volatility_state(df: pd.DataFrame, period: int = 14) -> Tuple[str, float]:
    a = atr(df, period=period).iloc[-1]
    c = df["close"].iloc[-1]
    if c == 0 or pd.isna(a):
        return "Unknown", float("nan")
    pct = float(a / c)
    if pct > 0.05:
        return "High", pct
    if pct > 0.02:
        return "Moderate", pct
    return "Low", pct


# ============================================================
# WHALE FLAGS
# ============================================================
# =========================
# QUIET ACCUMULATION
# =========================

def detect_quiet_accumulation(
    df: pd.DataFrame,
    rvol: float,
    atr_pct: float,
    trend: str,
    whale_count: int,
    cfg: VolumeTierConfig
) -> bool:
    """
    Detects low-volume, low-volatility compression
    that often precedes expansion.
    NOT a trade signal.
    """

    # Avoid aggressive phases
    if whale_count >= 2:
        return False

    # Must be below active volume threshold
    if rvol >= cfg.tier_b_min_rvol:
        return False

    # ATR compression threshold (~0.8%)
    if atr_pct > 0.008:
        return False

    # Avoid strong counter-trend
    if trend not in ("Neutral", "Bullish", "Bearish"):
        return False

    return True

def whale_flags(df: pd.DataFrame, cfg: WhaleFlagConfig) -> Dict[str, bool]:
    """
    Returns whale activity flags.
    These are heuristics designed for paper testing.
    """
    rvol_val = relative_volume(df)

    # ATR fall check
    atr_series = atr(df, period=14)
    idx_prev = -cfg.atr_fall_lookback
    if len(atr_series) < abs(idx_prev):
        atr_now = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        atr_prev = atr_now
    else:
        atr_now = float(atr_series.iloc[-1])
        atr_prev = float(atr_series.iloc[idx_prev])

    atr_fell = (atr_prev > 0) and (atr_now <= atr_prev * (1.0 - cfg.atr_fall_pct))
    stealth = (rvol_val >= cfg.stealth_rvol_min) and atr_fell

    # Absorption: tight range + baseline vol
    n = cfg.compression_window
    if len(df) >= n:
        recent = df.iloc[-n:]
        rng = float(recent["high"].max() - recent["low"].min())
        close = float(df["close"].iloc[-1]) if df["close"].iloc[-1] != 0 else 1.0
        compression = (rng / close) <= cfg.compression_pct
    else:
        compression = False
    absorption = compression and (rvol_val >= cfg.absorption_rvol_min)

    # Wick defense: repeated long lower wicks + vol not collapsing
    m = cfg.wick_checks
    wick_def = False
    if len(df) >= m:
        recent = df.iloc[-m:]
        bodies = (recent["close"] - recent["open"]).abs().replace(0, np.nan)
        lower_wicks = (recent[["open", "close"]].min(axis=1) - recent["low"]).clip(lower=0)
        ratio = (lower_wicks / bodies).replace([np.inf, -np.inf], np.nan)
        wick_count = int((ratio >= cfg.lower_wick_ratio_min).sum())
        wick_def = (wick_count >= max(2, m // 2)) and (rvol_val >= cfg.wick_vol_floor)

    return {
        "stealth_accumulation": bool(stealth),
        "absorption": bool(absorption),
        "wick_defense": bool(wick_def),
    }


# ============================================================
# TIER ENGINE (volume relaxed, whale-aware)
# ============================================================

def classify_tier(trend: str, rvol_val: float, whale_count: int, cfg: VolumeTierConfig) -> TierType:
    # Ignore non-directional markets
    if trend not in ("Bullish", "Bearish"):
        return "C"

    # Tier A: confirmed momentum
    if rvol_val >= cfg.tier_a_rvol:
        return "A"

    # NEW RULE:
    # Promote early whale setups to B+ if trend aligns
    if whale_count == 1 and rvol_val >= cfg.tier_b_min_rvol:
        return "B+"

    # Original B+ rule (stronger confirmation)
    if whale_count >= 2 and rvol_val >= cfg.tier_bplus_min_rvol:
        return "B+"

    # Tier B: early / signal-style
    if rvol_val >= cfg.tier_b_min_rvol:
        return "B"

    return "C"



# ============================================================
# RISK MANAGER
# ============================================================

def rr_ratio(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0
    return round(reward / risk, 2)


def position_size(bankroll: float, risk_per_trade: float, entry: float, stop: float, round_shares: bool = True) -> float:
    risk_amount = bankroll * risk_per_trade
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    size = risk_amount / dist
    if round_shares:
        return float(math.floor(size))
    return float(size)


def atr_stop_suggestion(df: pd.DataFrame, entry: float, direction: str, risk_cfg: RiskConfig) -> float:
    """
    Suggests a stop based on ATR.
    direction: 'Bullish' or 'Bearish'
    """
    a = atr(df, period=risk_cfg.atr_period).iloc[-1]
    if pd.isna(a) or a <= 0:
        return entry  # can't compute
    if direction == "Bullish":
        return float(entry - risk_cfg.atr_stop_mult * a)
    if direction == "Bearish":
        return float(entry + risk_cfg.atr_stop_mult * a)
    return entry


# ============================================================
# SCORING ENGINE (keeps the "original tool" flavor)
# ============================================================

@dataclass
class ScoreWeights:
    trend_bullish: int = 30
    trend_bearish: int = 20
    rr_good: int = 25
    volume_confirm: int = 25
    whale_bonus: int = 10


def score_setup(
    trend: str,
    rvol_val: float,
    rr: float,
    whale_count: int,
    tier: TierType,
    weights: ScoreWeights,
    vol_cfg: VolumeTierConfig,
    risk_cfg: RiskConfig
) -> Tuple[int, str]:
    """
    Produces a numeric score and a decision string.
    This keeps the classic "score/skip" style from your earlier tools,
    but now volume is tiered instead of vetoing.
    """
    score = 0

    if trend == "Bullish":
        score += weights.trend_bullish
    elif trend == "Bearish":
        score += weights.trend_bearish

    if rr >= risk_cfg.min_rr:
        score += weights.rr_good

    # Volume confirmation only if it's Tier A level
    if rvol_val >= vol_cfg.tier_a_rvol:
        score += weights.volume_confirm

    if whale_count >= 2:
        score += weights.whale_bonus

    # Decision mapping (tier + score)
    if tier == "A" and score >= 70:
        return score, "High-Quality (Confirmed)"
    if tier in ("B+", "B") and score >= 50:
        return score, "Early / Signal-Style (Paper-Test Friendly)"
    if tier == "C":
        return score, "Watch Only"
    return score, "Skip / Low Edge"


# ============================================================
# OUTPUT FORMATTERS (keep ticker visible, nice CLI prints)
# ============================================================

def format_flags(flags: Dict[str, bool]) -> str:
    on = [k for k, v in flags.items() if v]
    return ", ".join(on) if on else "None"


def pretty_print_result(res: Dict[str, object]) -> None:
    print("\n" + "=" * 68)
    print(f"SYMBOL: {res.get('symbol')}    TIME: {res.get('timestamp')}")
    print("-" * 68)
    print(f"Trend: {res.get('trend')}    Tier: {res.get('tier')}    Volatility: {res.get('vol_state')} (ATR% {res.get('atr_pct')})")
    print(f"RVOL: {res.get('rvol')}    WhaleFlags({res.get('whale_count')}): {format_flags(res.get('whale_flags', {}))}")
    print(f"Entry: {res.get('entry')}    Stop: {res.get('stop')}    Target: {res.get('target')}")
    print(f"RR: {res.get('rr')}    PositionSize: {res.get('position_size')}    Score: {res.get('score')}")
    print(f"Decision: {res.get('decision')}    PaperMode: {res.get('paper_mode')}")
    print("=" * 68 + "\n")


# ============================================================
# LOGGING (paper testing)
# ============================================================

def ensure_log_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_csv(path: str, row: Dict[str, object]) -> None:
    ensure_log_dir(os.path.dirname(path) or ".")
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        # Convert complex fields
        clean = {}
        for k, v in row.items():
            if isinstance(v, (dict, list, tuple)):
                clean[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean[k] = v
        w.writerow(clean)


# ============================================================
# EVALUATION CORE (manual + scout share this)
# ============================================================

def evaluate_symbol(
    provider: BaseProvider,
    symbol: str,
    tool_cfg: ToolConfig,
    entry: float,
    stop: float,
    target: float,
) -> Dict[str, object]:
    """
def evaluate_symbol(...):
    """
    # Full evaluation for a symbol with entry/stop/target.

    """
    df = provider.fetch_ohlcv(
        symbol,
        timeframe=tool_cfg.timeframe,
        limit=tool_cfg.candles
    )

    rvol = relative_volume(df)
    atr_pct = atr_pct_from_df(df)
    trend = trend_state(df)

    flags = whale_flags(df, cfg.whale)
    whale_count = sum(flags.values())

    quiet_accumulation = detect_quiet_accumulation(
        df=df,
        rvol=rvol,
        atr_pct=atr_pct,
        trend=trend,
        whale_count=whale_count,
        cfg=cfg.volume
    )

    # continue evaluation logic here

)

    tr = trend_state(df)
    vol_state, atr_pct = volatility_state(df, period=tool_cfg.risk.atr_period)
    rvol_val = relative_volume(df)
    flags = whale_flags(df, tool_cfg.whale)
    whale_count = int(sum(flags.values()))
    tier = classify_tier(tr, rvol_val, whale_count, tool_cfg.volume)

    # If stop not provided (0), suggest ATR stop
    actual_stop = stop
    if tool_cfg.risk.use_atr_stop and (stop == 0.0 or stop is None):
        actual_stop = atr_stop_suggestion(df, entry=entry, direction=tr, risk_cfg=tool_cfg.risk)

    rr = rr_ratio(entry, actual_stop, target)
    size = position_size(
        bankroll=tool_cfg.risk.bankroll,
        risk_per_trade=tool_cfg.risk.risk_per_trade,
        entry=entry,
        stop=actual_stop,
        round_shares=tool_cfg.risk.round_shares,
    )

    weights = ScoreWeights()
    score, decision = score_setup(
        trend=tr,
        rvol_val=rvol_val,
        rr=rr,
        whale_count=whale_count,
        tier=tier,
        weights=weights,
        vol_cfg=tool_cfg.volume,
        risk_cfg=tool_cfg.risk,
    )

    res = {
        "symbol": symbol,
        "timestamp": now_cst().isoformat(timespec="seconds"),
        "trend": tr,
        "tier": tier,
        "vol_state": vol_state,
        "atr_pct": (round(atr_pct, 4) if not pd.isna(atr_pct) else None),
        "rvol": round(rvol_val, 2),
        "whale_flags": flags,
        "whale_count": whale_count,
        "entry": float(entry),
        "stop": float(actual_stop),
        "target": float(target),
        "rr": rr,
        "position_size": size,
        "score": int(score),
        "decision": decision,
        "paper_mode": bool(tool_cfg.paper_mode),
        "quiet_accumulation": quiet_accumulation,

    }
    return res


def scout_symbol_quick(provider: BaseProvider, symbol: str, tool_cfg: ToolConfig) -> Dict[str, object]:
    """
    # Scout mode: tier + whale + trend without entry/stop/target.

    """
    df = provider.fetch_ohlcv(symbol, timeframe=tool_cfg.timeframe, limit=tool_cfg.candles)
    tr = trend_state(df)
    vol_state, atr_pct = volatility_state(df, period=tool_cfg.risk.atr_period)
    rvol_val = relative_volume(df)
    flags = whale_flags(df, tool_cfg.whale)
    whale_count = int(sum(flags.values()))
    tier = classify_tier(tr, rvol_val, whale_count, tool_cfg.volume)

    return {
        "symbol": symbol,
        "timestamp": now_cst().isoformat(timespec="seconds"),
        "trend": tr,
        "tier": tier,
        "vol_state": vol_state,
        "atr_pct": (round(atr_pct, 4) if not pd.isna(atr_pct) else None),
        "rvol": round(rvol_val, 2),
        "whale_flags": flags,
        "whale_count": whale_count,
        "decision": ("ALERT" if tier in ("A", "B+") else "LOG"),
        "paper_mode": bool(tool_cfg.paper_mode),
        "quiet_accumulation": quiet_accumulation,

    }


# ============================================================
# AUTO-SCOUT LOOP (no APScheduler)
# ============================================================

def should_run_scout(now: datetime, cfg: ScoutScheduleConfig, last_run: Optional[datetime]) -> bool:
    if not cfg.enabled:
        return False
    if not within_schedule_window(now, cfg):
        return False
    if not is_top_of_hour(now, cfg):
        return False
    if last_run is None:
        return True
    return (now - last_run).total_seconds() >= cfg.min_seconds_between_runs


def run_auto_scout_loop(tool_cfg: ToolConfig) -> None:
    """
    # Runs forever (worker mode). Every hour on the hour, during schedule window, scans watchlist.

    #This is how you run it on Railway as a WORKER service too.
    """
    provider = build_provider(tool_cfg.market)

    print("[AUTO] Auto-scout loop started.")
    print(f"[AUTO] Market={tool_cfg.market} Timeframe={tool_cfg.timeframe} Window=Mon-Fri {tool_cfg.schedule.start_hour}:00–{tool_cfg.schedule.end_hour}:00 CST")
    print(f"[AUTO] Watchlist size={len(tool_cfg.watchlist)}")
    print("[AUTO] Paper mode is ON." if tool_cfg.paper_mode else "[AUTO] Paper mode is OFF (still no execution in this script).")

    ensure_log_dir(tool_cfg.log_dir)
    scout_csv = os.path.join(tool_cfg.log_dir, tool_cfg.scout_log_csv)

    last_run: Optional[datetime] = None

    while True:
        now = now_cst()

        try:
            if should_run_scout(now, tool_cfg.schedule, last_run):
                stamp = now.strftime("%Y-%m-%d %H:%M:%S")
                print("\n" + "=" * 72)
                print(f"[AUTO] SCOUT RUN @ {stamp} CST")
                print("=" * 72)

                results: List[Dict[str, object]] = []
                for sym in tool_cfg.watchlist:
                    try:
                        r = scout_symbol_quick(provider, sym, tool_cfg)
                        results.append(r)
                    except Exception as e:
                        results.append({
                            "symbol": sym,
                            "timestamp": now.isoformat(timespec="seconds"),
                            "error": str(e),
                            "decision": "ERROR",
                        })

                # Rank: Tier A > B+ > B > C, then whale_count, then rvol
                tier_rank = {"A": 3, "B+": 2, "B": 1, "C": 0}
                def sort_key(x):
                    t = x.get("tier", "C")
                    return (
                        tier_rank.get(t, 0),
                        int(x.get("whale_count", 0) or 0),
                        float(x.get("rvol", 0) or 0),
                    )
                results.sort(key=sort_key, reverse=True)

                # Print top 10
                print(f"{'SYM':>8} | {'Tier':>4} | {'Trend':>7} | {'RVOL':>5} | {'Whale':>5} | {'Decision':>7}")
                print("-" * 72)
                for r in results[:10]:
                    if "error" in r:
                        print(f"{r.get('symbol', ''):>8} | {'-':>4} | {'-':>7} | {'-':>5} | {'-':>5} | ERROR")
                    else:
                        print(f"{r['symbol']:>8} | {r['tier']:>4} | {r['trend']:>7} | {r['rvol']:>5} | {r['whale_count']:>5} | {r['decision']:>7}")

                # Alert if any A or B+
                if tool_cfg.play_alert_sound and any(r.get("tier") in ("A", "B+") for r in results if "tier" in r):
                    cash_register_alert()

                # Log all results
                for r in results:
                    # flatten for CSV
                    row = dict(r)
                    if isinstance(row.get("whale_flags"), dict):
                        row["whale_flags"] = json.dumps(row["whale_flags"])
                    append_csv(scout_csv, row)

                last_run = now

        except Exception:
            # Never die in the loop; log and keep going
            print("[AUTO] Exception in auto loop:")
            traceback.print_exc()

        # Sleep a bit; check twice per minute
        time.sleep(30)


# ============================================================
# MANUAL CLI MENU (your "original tool" vibe)
# ============================================================

def prompt_float(msg: str, default: Optional[float] = None) -> float:
    while True:
        raw = input(msg).strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def manual_eval_flow(tool_cfg: ToolConfig) -> None:
    provider = build_provider(tool_cfg.market)

    print("\n[MANUAL] Manual evaluator")
    symbol = input("Enter symbol (e.g., AAPL or BTC/USDT or BTC): ").strip().upper()
    entry = prompt_float("Entry price: ")
    stop = prompt_float("Stop price (0 to auto ATR stop): ", default=0.0)
    target = prompt_float("Target price: ")

    try:
        res = evaluate_symbol(provider, symbol, tool_cfg, entry=entry, stop=stop, target=target)
        pretty_print_result(res)

        # Log manual eval
        ensure_log_dir(tool_cfg.log_dir)
        eval_csv = os.path.join(tool_cfg.log_dir, tool_cfg.eval_log_csv)
        append_csv(eval_csv, res)

    except Exception as e:
        print(f"[MANUAL] Error evaluating {symbol}: {e}")
        traceback.print_exc()


def print_config(tool_cfg: ToolConfig) -> None:
    print("\nCURRENT CONFIG")
    print("-" * 60)
    d = asdict(tool_cfg)
    # dataclasses nested -> already dict
    print(json.dumps(d, indent=2))
    print("-" * 60)


def cli_main(tool_cfg: ToolConfig) -> None:
    """
    #Classic CLI main loop.
    """
    while True:
        print("\n" + "=" * 72)
        print("TRADE EVALUATOR TOOL — REBUILD (Original-Style + Enhancements)")
        print("=" * 72)
        print(f"Market: {tool_cfg.market} | Timeframe: {tool_cfg.timeframe} | Candles: {tool_cfg.candles} | PaperMode: {tool_cfg.paper_mode}")
        print("1) Manual evaluate a symbol")
        print("2) Run ONE scout scan now (watchlist)")
        print("3) Start auto-scout loop (hourly M–F 7a–8p CST)")
        print("4) Show config")
        print("5) Exit")
        choice = input("Choose: ").strip()

        if choice == "1":
            manual_eval_flow(tool_cfg)

        elif choice == "2":
            provider = build_provider(tool_cfg.market)
            ensure_log_dir(tool_cfg.log_dir)
            scout_csv = os.path.join(tool_cfg.log_dir, tool_cfg.scout_log_csv)

            print("\n[SCOUT] Running one scan now...")
            results = []
            for sym in tool_cfg.watchlist:
                try:
                    r = scout_symbol_quick(provider, sym, tool_cfg)
                    results.append(r)
                except Exception as e:
                    results.append({"symbol": sym, "error": str(e), "decision": "ERROR"})

            tier_rank = {"A": 3, "B+": 2, "B": 1, "C": 0}
            results.sort(key=lambda x: (
                tier_rank.get(x.get("tier", "C"), 0),
                int(x.get("whale_count", 0) or 0),
                float(x.get("rvol", 0) or 0),
            ), reverse=True)

            print(f"{'SYM':>8} | {'Tier':>4} | {'Trend':>7} | {'RVOL':>5} | {'Whale':>5} | {'Decision':>7}")
            print("-" * 72)
            for r in results[:15]:
                if "error" in r:
                    print(f"{r.get('symbol',''):>8} | {'-':>4} | {'-':>7} | {'-':>5} | {'-':>5} | ERROR")
                else:
                    print(f"{r['symbol']:>8} | {r['tier']:>4} | {r['trend']:>7} | {r['rvol']:>5} | {r['whale_count']:>5} | {r['decision']:>7}")

            if tool_cfg.play_alert_sound and any(r.get("tier") in ("A", "B+") for r in results if "tier" in r):
                cash_register_alert()

            for r in results:
                row = dict(r)
                if isinstance(row.get("whale_flags"), dict):
                    row["whale_flags"] = json.dumps(row["whale_flags"])
                append_csv(scout_csv, row)

        elif choice == "3":
            run_auto_scout_loop(tool_cfg)

        elif choice == "4":
            print_config(tool_cfg)

        elif choice == "5":
            print("Bye.")
            return

        else:
            print("Invalid choice.")


# ============================================================
# OPTIONAL: FLASK SERVER MODE (Railway Web Service)
# ============================================================

def build_flask_app(tool_cfg: ToolConfig):
    # If you want this to run as a Railway WEB service, you need a server.
    # This provides endpoints to:
    #   - /health
    #   - /scout (one scan)
    #   - /eval  (manual eval via JSON)


    if Flask is None:
        raise RuntimeError("Flask is not installed. Add flask to requirements.txt")

    app = Flask(__name__)
    provider = build_provider(tool_cfg.market)

    @app.get("/")
    def home():
        return "Trade Evaluator Tool is running."

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "time": now_cst().isoformat(timespec="seconds"),
            "market": tool_cfg.market,
            "paper_mode": tool_cfg.paper_mode,
        })

    @app.get("/scout")
    def scout_once():
        results = []
        for sym in tool_cfg.watchlist:
            try:
                results.append(scout_symbol_quick(provider, sym, tool_cfg))
            except Exception as e:
                results.append({"symbol": sym, "error": str(e), "decision": "ERROR"})
        return jsonify(results)

    @app.get("/eval")
    def eval_get():
        symbol = flask_request.args.get("symbol", "").upper().strip()
        entry = float(flask_request.args.get("entry", 0))
        stop = float(flask_request.args.get("stop", 0))
        target = float(flask_request.args.get("target", 0))
    
        if not symbol or entry == 0 or target == 0:
            return jsonify({
                "error": "Usage: /eval?symbol=BTC&entry=42500&stop=41200&target=46500"
            }), 400
    
        try:
            res = evaluate_symbol(
                provider,
                symbol,
                tool_cfg,
                entry=entry,
                stop=stop,
                target=target
            )
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    return app


# ============================================================
# RAILWAY NOTES (IMPORTANT)
# ============================================================
"""
#Railway has two typical service styles:
#1) Web service (expects a port to be bound)
#2) Worker service (background loop / job)

#This script supports both:

#A) WEB SERVICE MODE:
   #    - Set env RUN_MODE=server
   #    - Ensure flask is installed
   #    - Start command: python app.py
   #    - Railway will provide PORT; we bind to it.

#B) WORKER MODE:
   #    - Set env RUN_MODE=worker
   #    - Start command: python app.py
   #    - This will run the auto-scout loop forever.
   #    - There is no web page in worker mode.

#If you tried to deploy CLI to a web service, it appears "online" but you see nothing.
#So you must choose server or worker mode.
"""


# ============================================================
# ENTRY POINT
# ============================================================

def load_tool_config_from_env() -> ToolConfig:
    """
    Allows you to override key settings via env vars.
    """
    cfg = ToolConfig()

    m = os.environ.get("MARKET")
    if m in ("stock", "crypto"):
        cfg.market = m  # type: ignore

    tf = os.environ.get("TIMEFRAME")
    if tf:
        cfg.timeframe = tf

    candles = os.environ.get("CANDLES")
    if candles and candles.isdigit():
        cfg.candles = int(candles)

    paper = os.environ.get("PAPER_MODE")
    if paper is not None:
        cfg.paper_mode = paper.strip().lower() in ("1", "true", "yes", "y")

    alert = os.environ.get("ALERT_SOUND")
    if alert is not None:
        cfg.play_alert_sound = alert.strip().lower() in ("1", "true", "yes", "y")

    # watchlist override (comma-separated)
    wl = os.environ.get("WATCHLIST")
    if wl:
        parts = [x.strip().upper() for x in wl.split(",") if x.strip()]
        if parts:
            cfg.watchlist = tuple(parts)

    return cfg

if __name__ == "__main__":
    tool_cfg = load_tool_config_from_env()

    run_mode = os.environ.get("RUN_MODE", "").strip().lower()

    # HARD STOP: Railway must NEVER run CLI
    on_railway = bool(
        os.environ.get("RAILWAY_PROJECT_ID")
        or os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("RAILWAY_SERVICE_ID")
    )

    if on_railway and run_mode not in ("server", "worker"):
        print("[FATAL] CLI mode is disabled on Railway.")
        print("Set RUN_MODE=server or RUN_MODE=worker.")
        sys.exit(1)

    if run_mode == "server":
        app = build_flask_app(tool_cfg)
        port = int(os.environ.get("PORT", "8080"))
        print("[BOOT] Server mode starting on port", port)
        app.run(host="0.0.0.0", port=port)

    elif run_mode == "worker":
        print("[BOOT] Worker mode starting (auto-scout loop)")
        run_auto_scout_loop(tool_cfg)

    else:
        # LOCAL ONLY — never reachable on Railway
        print("[BOOT] Local CLI mode")
        cli_main(tool_cfg)
"""

        



   


  

