"""RSI (Relative Strength Index) analysis module"""

from typing import Dict

def analyze_rsi(rsi: float) -> Dict[str, str]:
    """Analyze RSI value and determine trend"""
    if rsi < 30:
        return {"trend": "strongly_oversold"}
    elif rsi < 40:
        return {"trend": "oversold"}
    elif rsi > 70:
        return {"trend": "strongly_overbought"}
    elif rsi > 60:
        return {"trend": "overbought"}
    else:
        return {"trend": "neutral"}
