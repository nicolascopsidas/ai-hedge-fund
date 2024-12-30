"""Configuration settings for the AI Trading Bot"""

from datetime import timedelta
from typing import Dict, Any

# API Configuration
POLYGON_BASE_URL_V1 = "https://api.polygon.io/v1"
POLYGON_BASE_URL_V2 = "https://api.polygon.io/v2"

# Timeframe Settings
MAX_STALENESS: Dict[str, timedelta] = {
    "minute": timedelta(minutes=30),
    "hour": timedelta(hours=4),
    "day": timedelta(days=1)
}

# Technical Indicator Settings
INDICATOR_SETTINGS = {
    "rsi": {
        "period": 14
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    },
    "ema": {
        "short_period": 20,
        "long_period": 50
    },
    "sma": {
        "short_period": 20,
        "long_period": 50
    }
}

# Analysis Settings
VOLATILITY_THRESHOLDS = {
    "low": 0.5,
    "medium": 1.5,
    "high": 3.0
}

RSI_THRESHOLDS = {
    "oversold": 30,
    "overbought": 70
}
