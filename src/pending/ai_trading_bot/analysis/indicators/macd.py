"""MACD (Moving Average Convergence Divergence) analysis module"""

from typing import Dict
import pandas as pd

def analyze_macd(macd_data: Dict) -> Dict[str, any]:
    """Analyze MACD indicator with better trend detection"""
    if not isinstance(macd_data, dict):
        return {
            "trend": "neutral",
            "current_histogram": {
                "value": 0,
                "description": "Valeur actuelle de l'histogramme MACD"
            },
            "evolution_histogram": {
                "value": 0,
                "description": "Evolution de l'histogramme du MACD"
            },
            "recent_histogram": {
                "values": [0] * 5,
                "description": "Valeurs recentes de l'histogramme MACD"
            },
            "strength": "weak"
        }

    # Get histogram values
    histogram_data = macd_data.get("histogram", None)
    if isinstance(histogram_data, pd.Series) and not histogram_data.empty:
        recent_values = histogram_data.iloc[-5:].tolist() if len(histogram_data) >= 5 else [histogram_data.iloc[-1]] * 5
        current_value = histogram_data.iloc[-1]
        evolution = current_value - histogram_data.iloc[-2] if len(histogram_data) > 1 else 0
    else:
        recent_values = [0] * 5
        current_value = 0
        evolution = 0

    # Determine trend
    if current_value > 0:
        trend = "bullish"
        strength = "strong" if evolution > 0 else "weak"
    elif current_value < 0:
        trend = "bearish"
        strength = "strong" if evolution < 0 else "weak"
    else:
        trend = "neutral"
        strength = "weak"

    return {
        "trend": trend,
        "current_histogram": {
            "value": round(current_value, 5),
            "description": "Valeur actuelle de l'histogramme MACD"
        },
        "evolution_histogram": {
            "value": round(evolution, 5),
            "description": "Evolution de l'histogramme du MACD"
        },
        "recent_histogram": {
            "values": [round(x, 5) for x in recent_values],
            "description": "Valeurs recentes de l'histogramme MACD"
        },
        "strength": strength
    }
