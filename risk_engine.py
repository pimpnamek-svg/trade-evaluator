from dataclasses import dataclass

@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    breakeven_at: float | None = None


def atr_risk_engine(
    atr: float,
    volatility_mode: str = "normal"  # "low", "normal", "high"
) -> RiskDecision:
    """
    Enforces Gizmo Sticky-Note Rules
    """

    # === HARD FILTERS ===
    if atr < 0.18:
        return RiskDecision(False, "ATR too low (dead market)")

    if atr > 0.45:
        return RiskDecision(False, "ATR too high (chaotic volatility)")

    # === SL MULTIPLIER BY MODE ===
    sl_multipliers = {
        "low": 0.8,
        "normal": 1.2,
        "high": 1.0,
    }

    sl_multiplier = sl_multipliers.get(volatility_mode, 1.2)

    raw_sl = atr * sl_multiplier
    stop_loss = min(raw_sl, 0.45)

    if stop_loss >= 0.45:
        return RiskDecision(False, "Stop loss exceeds max 0.45")

    # === TAKE PROFITS ===
    tp1 = atr * 1.0
    tp2 = atr * 1.8

    # === RISK / REWARD CHECK ===
    if tp1 < stop_loss * 0.8:
        return RiskDecision(False, "Risk/Reward below minimum threshold")

    # === BREAKEVEN RULE ===
    breakeven_at = atr * 0.5

    return RiskDecision(
        allowed=True,
        reason="Trade approved by Gizmo",
        stop_loss=round(stop_loss, 3),
        tp1=round(tp1, 3),
        tp2=round(tp2, 3),
        breakeven_at=round(breakeven_at, 3),
    )
