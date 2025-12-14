# app.py
# Railway-compatible Trade Evaluator
# Engine + Minimal Flask Server
# Paper trading ONLY

import os
from flask import Flask, jsonify
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

# =========================
# CONFIG
# =========================

@dataclass
class TradeConfig:
    bankroll: float = 3000.0
    risk_per_trade: float = 0.03
    min_rr: float = 2.0
    paper_mode: bool = True

    tier_a_rvol: float = 1.50
    tier_bplus_min: float = 1.05
    tier_b_min: float = 0.80
    whale_vol_min: float = 1.10


# =========================
# CORE ENGINE
# =========================

class TradeEvaluator:

    def __init__(self, cfg: TradeConfig):
        self.cfg = cfg

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
        avg = df["volume"].rolling(period).mean().iloc[-1]
        if avg == 0 or pd.isna(avg):
            return 0.0
        return round(df["volume"].iloc[-1] / avg, 2)

    def trend_state(self, df: pd.DataFrame) -> str:
        sma30 = self.sma(df["close"], 30)
        sma50 = self.sma(df["close"], 50)
        if sma30.iloc[-1] > sma50.iloc[-1]:
            return "Bullish"
        if sma30.iloc[-1] < sma50.iloc[-1]:
            return "Bearish"
        return "Neutral"

    def whale_flags(self, df: pd.DataFrame) -> Dict[str, bool]:
        rvol = self.relative_volume(df)
        atr_series = self.atr(df)
        atr_now = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-6] if len(atr_series) > 6 else atr_now

        return {
            "stealth_accumulation": rvol >= self.cfg.whale_vol_min and atr_now <= atr_prev,
            "absorption": rvol >= 1.0
        }

    def classify_tier(self, trend: str, rvol: float, whale_count: int) -> str:
        if trend not in ("Bullish", "Bearish"):
            return "C"
        if rvol >= self.cfg.tier_a_rvol:
            return "A"
        if rvol >= self.cfg.tier_bplus_min and whale_count >= 1:
            return "B+"
        if rvol >= self.cfg.tier_b_min:
            return "B"
        return "C"

    def evaluate_trade(self, df, entry, stop, target, symbol):
        trend = self.trend_state(df)
        rvol = self.relative_volume(df)
        whales = self.whale_flags(df)
        tier = self.classify_tier(trend, rvol, sum(whales.values()))

        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = round(reward / risk, 2) if risk else 0.0

        decision = "IGNORE"
        if rr >= self.cfg.min_rr and tier in ("A", "B+", "B"):
            decision = "VALID SETUP"

        return {
            "symbol": symbol,
            "time": datetime.now().isoformat(timespec="seconds"),
            "trend": trend,
            "tier": tier,
            "relative_volume": rvol,
            "whale_flags": whales,
            "rr": rr,
            "decision": decision,
            "paper_mode": self.cfg.paper_mode
        }


# =========================
# FLASK SERVER (REQUIRED)
# =========================

app = Flask(__name__)


   


  

