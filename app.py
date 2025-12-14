# app.py
# ORIGINAL Trade Evaluator + Tiered Volume + Whale Flags
# MANUAL EVALUATION — STABLE VERSION

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

    # volume thresholds
    tier_a_rvol: float = 1.50
    tier_bplus_min: float = 1.05
    tier_b_min: float = 0.80

    # whale logic
    whale_vol_min: float = 1.10


# =========================
# CORE EVALUATOR
# =========================

class TradeEvaluator:

    def __init__(self, cfg: TradeConfig):
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
        tr = pd.concat(
            [high_low, high_close, low_close],
            axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def relative_volume(df: pd.DataFrame, period: int = 20) -> float:
        avg_vol = df["volume"].rolling(period).mean().iloc[-1]
        if avg_vol == 0 or np.isnan(avg_vol):
            return 0.0
        return round(df["volume"].iloc[-1] / avg_vol, 2)


   


  

