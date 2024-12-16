from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import pandas as pd


class TimeFrame(Enum):
    M5 = "5min"    # 5 minutes - Quick signals, noise filtering
    M15 = "15min"  # 15 minutes - Entry/exit confirmation
    H1 = "1hour"   # 1 hour - Primary trend direction
    H4 = "4hour"   # 4 hours - Major support/resistance
    D1 = "1day"    # 1 day - Overall market context
    
    @classmethod
    def get_lookback(cls, timeframe: 'TimeFrame') -> timedelta:
        """Get the lookback period for each timeframe"""
        lookback_map = {
            cls.M5: timedelta(hours=24),    # 24h of 5m data
            cls.M15: timedelta(days=3),     # 3 days of 15m data
            cls.H1: timedelta(days=7),      # 7 days of 1h data
            cls.H4: timedelta(days=30),     # 30 days of 4h data
            cls.D1: timedelta(days=90),     # 90 days of daily data
        }
        return lookback_map[timeframe]
    
    @classmethod
    def from_string(cls, value: str) -> 'TimeFrame':
        """Convert string to TimeFrame enum"""
        for tf in cls:
            if tf.value == value:
                return tf
        raise ValueError(f"Invalid timeframe: {value}")
    
    @property
    def minutes(self) -> int:
        """Get the number of minutes for this timeframe"""
        minutes_map = {
            self.M5: 5,
            self.M15: 15,
            self.H1: 60,
            self.H4: 240,
            self.D1: 1440
        }
        return minutes_map[self]
    
    @property
    def next_timeframe(self) -> Optional['TimeFrame']:
        """Get the next larger timeframe"""
        timeframes = list(TimeFrame)
        current_index = timeframes.index(self)
        if current_index < len(timeframes) - 1:
            return timeframes[current_index + 1]
        return None
    
    @property
    def previous_timeframe(self) -> Optional['TimeFrame']:
        """Get the previous smaller timeframe"""
        timeframes = list(TimeFrame)
        current_index = timeframes.index(self)
        if current_index > 0:
            return timeframes[current_index - 1]
        return None


@dataclass
class MarketSignal:
    timeframe: TimeFrame
    direction: str  # 'long', 'short', or 'neutral'
    strength: float  # 0.0 to 1.0
    indicators: Dict[str, Any]
    timestamp: datetime


@dataclass
class TradeSetup:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    leverage: float
    expected_duration: timedelta
    timeframe_confluence: int
    signal_strength: float
    timestamp: datetime


@dataclass
class Position:
    trade_id: str
    direction: str
    entry_price: float
    current_price: float
    size: float
    leverage: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    expected_duration: timedelta
    pnl: float
    status: str  # 'open', 'closed', 'pending'
    
    @property
    def duration(self) -> timedelta:
        """Get the current duration of the trade"""
        return datetime.now() - self.entry_time
    
    @property
    def is_expired(self) -> bool:
        """Check if the trade has exceeded its expected duration"""
        return self.duration > self.expected_duration
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate the risk/reward ratio of the trade"""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0
