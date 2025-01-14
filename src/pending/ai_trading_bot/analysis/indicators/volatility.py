"""Volatility analysis module"""

from typing import Tuple

def calculate_volatility(high: float, low: float) -> Tuple[str, float]:
    """Calculate volatility level and percentage"""
    volatility = (high - low) / low * 100
    
    if volatility < 0.5:
        return "low", volatility
    elif volatility < 2.0:
        return "medium", volatility
    else:
        return "high", volatility
