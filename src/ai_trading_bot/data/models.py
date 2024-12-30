"""Data models for the AI Trading Bot"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class PriceData:
    """Price data for a specific timeframe"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None

@dataclass
class IndicatorValue:
    """A single indicator value with its timestamp"""
    timestamp: datetime
    value: float

@dataclass
class TimeframeData:
    """Complete data for a specific timeframe"""
    timeframe: str
    current_price: PriceData
    historical_prices: List[PriceData]
    rsi_values: List[IndicatorValue]
    macd_values: List[IndicatorValue]
    macd_signal: List[IndicatorValue]
    macd_histogram: List[IndicatorValue]
    ema_20: List[IndicatorValue]
    ema_50: List[IndicatorValue]
    sma_20: List[IndicatorValue]
    sma_50: List[IndicatorValue]
    volatility: float
