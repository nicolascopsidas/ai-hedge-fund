from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from src.short_term.models import TimeFrame, MarketSignal


class TechnicalAnalyzer:
    def __init__(self):
        """Initialize technical analysis indicators configuration."""
        self.indicators = {
            "fast": {  # For 5m and 15m
                "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bb": {"period": 20, "std": 2.0},
            },
            "medium": {  # For 1h and 4h
                "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "atr": {"period": 14},
            },
            "slow": {  # For 1d
                "ema": {"period": 50},
                "atr": {"period": 14},
                "volume_sma": {"period": 20},
            }
        }
    
    def analyze_timeframe(self, 
                         data: pd.DataFrame, 
                         timeframe: TimeFrame) -> MarketSignal:
        """Analyze a specific timeframe and generate signals."""
        try:
            if data.empty:
                raise ValueError(f"No data available for {timeframe.value}")
            
            # Validate required columns
            required_columns = ['close', 'volume', 'rsi_values', 'macd_values', 'sma_values', 'ema_values']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Determine which indicator set to use
            if timeframe in [TimeFrame.M5, TimeFrame.M15]:
                indicators = self.indicators["fast"]
                analysis = self._analyze_fast(data)
            elif timeframe in [TimeFrame.H1, TimeFrame.H4]:
                indicators = self.indicators["medium"]
                analysis = self._analyze_medium(data)
            else:  # D1
                indicators = self.indicators["slow"]
                analysis = self._analyze_slow(data)
            
            return MarketSignal(
                timeframe=timeframe,
                direction=analysis["direction"],
                strength=analysis["strength"],
                indicators=analysis["indicators"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error analyzing {timeframe.value}: {str(e)}")
            return MarketSignal(
                timeframe=timeframe,
                direction="neutral",
                strength=0.0,
                indicators={},
                timestamp=datetime.now()
            )
    
    def _calculate_trend_strength(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate trend strength using linear regression."""
        try:
            prices = data['close'].tail(window)
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            return min(abs(slope) / prices.mean(), 1.0)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> float:
        """Analyze volume profile for trend confirmation."""
        try:
            recent_volume = data['volume'].tail(5).mean()
            historical_volume = data['volume'].mean()
            return min(recent_volume / historical_volume, 2.0) / 2.0  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _analyze_fast(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fast timeframes (5m, 15m) using Polygon indicators."""
        try:
            # Get indicators from DataFrame
            rsi = data['rsi_values'].iloc[-1]
            macd = data['macd_values'].iloc[-1]
            sma = data['sma_values'].iloc[-1]
            ema = data['ema_values'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Calculate additional metrics
            trend_strength = self._calculate_trend_strength(data)
            volume_profile = self._calculate_volume_profile(data)
            
            # Generate signals with confidence levels
            rsi_signal = (1 if rsi < 30 else -1 if rsi > 70 else 0) * (abs(50 - rsi) / 50)
            macd_signal = (1 if macd > 0 else -1 if macd < 0 else 0) * min(abs(macd) / 100, 1.0)
            trend_signal = (1 if current_price > sma else -1 if current_price < sma else 0) * trend_strength
            
            # Weight the signals
            weights = {"rsi": 0.3, "macd": 0.3, "trend": 0.4}
            combined_signal = (
                rsi_signal * weights["rsi"] +
                macd_signal * weights["macd"] +
                trend_signal * weights["trend"]
            )
            
            # Determine direction and strength
            direction = "long" if combined_signal > 0.1 else \
                       "short" if combined_signal < -0.1 else "neutral"
            
            strength = min(abs(combined_signal), 1.0)  # Normalize to 0-1
            
            return {
                "direction": direction,
                "strength": strength,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "sma": sma,
                    "ema": ema,
                    "price": current_price,
                    "trend_strength": trend_strength,
                    "volume_profile": volume_profile
                }
            }
            
        except Exception as e:
            print(f"Error in fast analysis: {str(e)}")
            return {
                "direction": "neutral",
                "strength": 0.0,
                "indicators": {}
            }
    
    def _analyze_medium(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze medium timeframes (1h, 4h) using Polygon indicators."""
        try:
            # Get indicators
            rsi = data['rsi_values'].iloc[-1]
            macd = data['macd_values'].iloc[-1]
            sma = data['sma_values'].iloc[-1]
            ema = data['ema_values'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Calculate additional metrics
            trend_strength = self._calculate_trend_strength(data, window=40)
            volume_profile = self._calculate_volume_profile(data)
            
            # Generate signals with confidence levels
            rsi_signal = (1 if rsi < 35 else -1 if rsi > 65 else 0) * (abs(50 - rsi) / 50)
            macd_signal = (1 if macd > 0 else -1 if macd < 0 else 0) * min(abs(macd) / 200, 1.0)
            trend_signal = (1 if current_price > ema else -1 if current_price < ema else 0) * trend_strength
            
            # Weight the signals
            weights = {"rsi": 0.2, "macd": 0.4, "trend": 0.4}
            combined_signal = (
                rsi_signal * weights["rsi"] +
                macd_signal * weights["macd"] +
                trend_signal * weights["trend"]
            )
            
            # Determine direction and strength
            direction = "long" if combined_signal > 0.15 else \
                       "short" if combined_signal < -0.15 else "neutral"
            
            strength = min(abs(combined_signal), 1.0)
            
            return {
                "direction": direction,
                "strength": strength,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "sma": sma,
                    "ema": ema,
                    "price": current_price,
                    "trend_strength": trend_strength,
                    "volume_profile": volume_profile
                }
            }
            
        except Exception as e:
            print(f"Error in medium analysis: {str(e)}")
            return {
                "direction": "neutral",
                "strength": 0.0,
                "indicators": {}
            }
    
    def _analyze_slow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze slow timeframes (1d) using Polygon indicators."""
        try:
            # Get indicators
            rsi = data['rsi_values'].iloc[-1]
            macd = data['macd_values'].iloc[-1]
            sma = data['sma_values'].iloc[-1]
            ema = data['ema_values'].iloc[-1]
            current_price = data['close'].iloc[-1]
            volume = data['volume'].iloc[-1]
            vwap = data['vwap'].iloc[-1]
            
            # Calculate additional metrics
            trend_strength = self._calculate_trend_strength(data, window=50)
            volume_profile = self._calculate_volume_profile(data)
            
            # Generate signals with confidence levels
            trend_signal = (1 if current_price > ema else -1 if current_price < ema else 0) * trend_strength
            volume_signal = (1 if volume > volume.mean() else -1 if volume < volume.mean() else 0) * volume_profile
            vwap_signal = (1 if current_price > vwap else -1 if current_price < vwap else 0) * min(abs(current_price - vwap) / vwap, 1.0)
            
            # Weight the signals
            weights = {"trend": 0.5, "volume": 0.2, "vwap": 0.3}
            combined_signal = (
                trend_signal * weights["trend"] +
                volume_signal * weights["volume"] +
                vwap_signal * weights["vwap"]
            )
            
            # Determine direction and strength
            direction = "long" if combined_signal > 0.2 else \
                       "short" if combined_signal < -0.2 else "neutral"
            
            strength = min(abs(combined_signal), 1.0)
            
            return {
                "direction": direction,
                "strength": strength,
                "indicators": {
                    "rsi": rsi,
                    "macd": macd,
                    "sma": sma,
                    "ema": ema,
                    "volume": volume,
                    "vwap": vwap,
                    "price": current_price,
                    "trend_strength": trend_strength,
                    "volume_profile": volume_profile
                }
            }
            
        except Exception as e:
            print(f"Error in slow analysis: {str(e)}")
            return {
                "direction": "neutral",
                "strength": 0.0,
                "indicators": {}
            }
