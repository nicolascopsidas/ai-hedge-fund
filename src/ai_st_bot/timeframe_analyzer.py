from datetime import datetime
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ai_st_data import AISTDataFetcher


class TimeframeAnalyzer:
    """Analyze market data across multiple timeframes."""
    
    def __init__(self):
        self.fetcher = AISTDataFetcher()

    def _calculate_volatility(self, high: float, low: float) -> Tuple[str, float]:
        """Calculate volatility level."""
        volatility_pct = ((high - low) / low) * 100
        if volatility_pct > 5:
            return "high", volatility_pct
        elif volatility_pct > 2:
            return "moderate", volatility_pct
        return "low", volatility_pct

    def _get_rsi_trend(self, rsi: float) -> str:
        """Get trend based on RSI value."""
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        return "neutral"

    def _analyze_macd(self, macd_data: Dict) -> Dict:
        """Analyze MACD data."""
        if not macd_data:
            return {"error": "No MACD data available"}
        
        histogram = macd_data.get("histogram", [])
        if not histogram or len(histogram) < 1:
            return {"error": "Not enough MACD histogram data"}
        
        current_histogram = histogram[-1]
        return {
            "current_histogram": {"value": current_histogram, "description": "Current MACD histogram"},
            "trend": "bullish" if current_histogram > 0 else "bearish"
        }

    def _analyze_bollinger_bands(self, price_data: pd.DataFrame) -> Optional[dict]:
        """Analyze Bollinger Bands based on price data."""
        if 'close' not in price_data:
            return {"error": "Missing 'close' data for Bollinger Bands"}
        
        rolling_mean = price_data['close'].rolling(window=20).mean()
        rolling_std = price_data['close'].rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        last_price = price_data['close'].iloc[-1]
        position = "inside"
        if last_price > upper_band.iloc[-1]:
            position = "above"
        elif last_price < lower_band.iloc[-1]:
            position = "below"

        return {
            "upper_band": round(upper_band.iloc[-1], 6),
            "lower_band": round(lower_band.iloc[-1], 6),
            "position": position,
            "description": "Current Bollinger Bands analysis"
        }

    def _analyze_fibonacci_retracements(self, price_data: pd.DataFrame) -> Optional[dict]:
        """Analyze Fibonacci retracement levels based on price data."""
        if 'high' not in price_data or 'low' not in price_data:
            return {"error": "Missing 'high' or 'low' data for Fibonacci Retracements"}
        
        high = price_data['high'].max()
        low = price_data['low'].min()
        diff = high - low
        
        levels = {
            "0%": round(low, 6),
            "23.6%": round(high - diff * 0.236, 6),
            "38.2%": round(high - diff * 0.382, 6),
            "50%": round(high - diff * 0.5, 6),
            "61.8%": round(high - diff * 0.618, 6),
            "100%": round(high, 6)
        }

        return {
            "high": high,
            "low": low,
            "levels": levels,
            "description": "Fibonacci retracement levels"
        }

    def _detect_chart_patterns(self, price_data: pd.DataFrame) -> List[Dict]:
        """Detect common chart patterns such as head and shoulders, triangles, and double tops/bottoms."""
        detected_patterns = []

        if self._is_head_and_shoulders(price_data):
            detected_patterns.append({
                "pattern": "Head and Shoulders",
                "type": "bearish",
                "description": "Detected Head and Shoulders pattern, indicating potential reversal."
            })

        if self._is_double_top(price_data):
            detected_patterns.append({
                "pattern": "Double Top",
                "type": "bearish",
                "description": "Detected Double Top pattern, indicating potential resistance at the highs."
            })

        if self._is_triangle(price_data):
            detected_patterns.append({
                "pattern": "Triangle",
                "type": "neutral",
                "description": "Detected Triangle pattern, awaiting breakout."
            })

        return detected_patterns

    def _is_head_and_shoulders(self, price_data: pd.DataFrame) -> bool:
        """Detect if Head and Shoulders pattern is present."""
        if len(price_data) < 5:
            return False

        peaks = price_data['close'].rolling(window=5).apply(lambda x: x[2] > max(x[0], x[1], x[3], x[4]), raw=True)
        peak_indices = np.where(peaks)[0]

        return len(peak_indices) >= 3

    def _is_double_top(self, price_data: pd.DataFrame) -> bool:
        """Detect if Double Top pattern is present."""
        if len(price_data) < 5:
            return False

        peaks = price_data['close'].rolling(window=5).apply(lambda x: x[2] > max(x[0], x[1], x[3], x[4]), raw=True)
        peak_indices = np.where(peaks)[0]

        if len(peak_indices) >= 2:
            return abs(price_data['close'][peak_indices[0]] - price_data['close'][peak_indices[1]]) < 0.01
        return False

    def analyze_timeframe(self, ticker: str, timeframe: str) -> dict:
        """Analyze a specific timeframe."""
        config = {
            "5m": {"timespan": "minute", "multiplier": 5},
            "15m": {"timespan": "minute", "multiplier": 15},
            "1h": {"timespan": "hour", "multiplier": 1},
            "4h": {"timespan": "hour", "multiplier": 4},
            "1d": {"timespan": "day", "multiplier": 1}
        }[timeframe]
        
        print(f"Fetching data for {ticker} on {timeframe} timeframe...")
        price_data, indicators = self.fetcher.get_market_data(ticker, config['timespan'], config['multiplier'])
        
        if price_data.empty or not indicators:
            print("No data fetched. Exiting analysis.")
            return None
            
        print("Data fetched successfully.")
        
        latest_price = price_data['close'].iloc[-1]
        high = price_data['high'].max()
        low = price_data['low'].min()
        
        volatility_level, volatility_pct = self._calculate_volatility(high, low)
        
        rsi_data = indicators.get('rsi', {}).get('value', pd.Series())
        rsi = rsi_data.iloc[-1] if not rsi_data.empty else 50
        rsi_trend = self._get_rsi_trend(rsi)
        
        macd_analysis = self._analyze_macd(indicators.get('macd', {}))
        
        bollinger_analysis = self._analyze_bollinger_bands(price_data)
        fibonacci_analysis = self._analyze_fibonacci_retracements(price_data)
        chart_patterns = self._detect_chart_patterns(price_data)

        return {
            "timeframe": timeframe,
            "price": {
                "current": {
                    "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                    "value": round(float(latest_price), 6),
                    "description": "Current price"
                }
            },
            "trend": "bullish" if rsi_trend == "overbought" else "bearish",
            "indicators": {
                "RSI": {
                    "current": {
                        "value": round(rsi, 2),
                        "description": "Relative Strength Index (RSI)"
                    },
                    "trend": rsi_trend
                },
                "MACD": macd_analysis,
                "Bollinger_Bands": bollinger_analysis,
                "Fibonacci": fibonacci_analysis
            },
            "volatility": volatility_level,
            "chart_patterns": chart_patterns
        }
