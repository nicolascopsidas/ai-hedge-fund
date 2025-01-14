"""EMA (Exponential Moving Average) analysis module"""

from typing import Dict, List, Tuple, Any

import pandas as pd

def analyze_ema_alignment(ema_20: pd.Series, ema_50: pd.Series, price: float) -> Dict[str, Any]:
    """Analyze EMA alignment with current price"""
    if not isinstance(ema_20, pd.Series) or not isinstance(ema_50, pd.Series) or ema_20 is None or ema_50 is None or ema_20.empty or ema_50.empty:
        return {
            "trend": "neutral",
            "aligned_indicators": [],
            "ema_20": {
                "value": price,
                "description": "Exponential Moving Average (EMA) sur 20 periodes"
            },
            "ema_50": {
                "value": price,
                "description": "Exponential Moving Average (EMA) sur 50 periodes"
            }
        }
        
    current_ema_20 = float(ema_20.iloc[-1])
    current_ema_50 = float(ema_50.iloc[-1])
    
    aligned_indicators = []
    
    # Check price position relative to EMAs
    if price > current_ema_20:
        aligned_indicators.append("price_above_ema20")
    if price > current_ema_50:
        aligned_indicators.append("price_above_ema50")
        
    # Determine trend based on EMA alignment
    if current_ema_20 > current_ema_50:
        trend = "bullish"
        if price > current_ema_20:
            aligned_indicators.append("full_bull_alignment")
    elif current_ema_20 < current_ema_50:
        trend = "bearish"
        if price < current_ema_20:
            aligned_indicators.append("full_bear_alignment")
    else:
        trend = "neutral"
        
    return {
        "trend": trend,
        "aligned_indicators": aligned_indicators,
        "ema_20": {
            "value": round(current_ema_20, 2),
            "description": "Exponential Moving Average (EMA) sur 20 periodes"
        },
        "ema_50": {
            "value": round(current_ema_50, 2),
            "description": "Exponential Moving Average (EMA) sur 50 periodes"
        }
    }

def format_ema_data(ema_data: pd.Series, period: int) -> Dict[str, Any]:
    """Format EMA data for output"""
    if ema_data is None or ema_data.empty:
        return {
            "current": None,
            "recent_values": [],
            "period": period
        }
        
    current_value = float(ema_data.iloc[-1])
    recent_values = ema_data.iloc[-5:].tolist()
    
    return {
        "current": round(current_value, 5),
        "recent_values": [round(x, 5) for x in recent_values],
        "period": period
    }
