"""Single timeframe analysis module"""

from typing import Dict, List, Optional

import pandas as pd

from src.ai_trading_bot.analysis.indicators.rsi import analyze_rsi
from src.ai_trading_bot.analysis.indicators.macd import analyze_macd
from src.ai_trading_bot.analysis.indicators.ema import analyze_ema_alignment, format_ema_data
from src.ai_trading_bot.analysis.indicators.volatility import calculate_volatility
from src.ai_trading_bot.utils.formatting import format_price_data, format_number

class TimeframeAnalyzer:
    """Analyzes market data for a specific timeframe"""

    def analyze_timeframe(
        self,
        price_data: pd.DataFrame,
        indicators: Dict[str, Dict]
    ) -> Dict[str, any]:
        """Analyze market data for a specific timeframe"""
        if price_data.empty:
            return None
            
        # Get latest price and calculate volatility
        latest_price = price_data["close"].iloc[-1]
        high = price_data["high"].max()
        low = price_data["low"].min()
        
        volatility_level, volatility_pct = calculate_volatility(high, low)
        
        # Get RSI and trend
        rsi_data = indicators.get("rsi", {}).get("value")
        if isinstance(rsi_data, pd.Series) and not rsi_data.empty:
            rsi = rsi_data.iloc[-1]
            rsi_values = rsi_data.iloc[-5:].tolist()  # Get last 5 values
            rsi_evolution = rsi - rsi_data.iloc[-2] if len(rsi_data) > 1 else 0
        else:
            rsi = 50  # Default neutral value
            rsi_values = [50] * 5
            rsi_evolution = 0
            
        rsi_analysis = analyze_rsi(rsi)
        
        # Get MACD with improved analysis
        macd_analysis = analyze_macd(indicators.get("macd", {}))
        
        # Get EMAs and alignment
        ema_20 = indicators.get("ema20", {}).get("value")
        ema_50 = indicators.get("ema50", {}).get("value")
        
        ema_analysis = analyze_ema_alignment(ema_20, ema_50, latest_price)
        
        # Calculate trend strength
        trend_strength = self._get_trend_strength(
            rsi,
            macd_analysis["current_histogram"]["value"],
            ema_analysis["trend"]
        )
        
        # Get volume data
        volume = price_data["volume"].iloc[-1]
        volume_trend = "increasing" if price_data["volume"].iloc[-1] > price_data["volume"].iloc[-2] else "decreasing"
        
        # Format output
        return {
            "price": format_price_data(price_data),
            "volume": {
                "current": format_number(volume),
                "trend": volume_trend,
                "vwap": format_number(price_data["vwap"].iloc[-1]) if "vwap" in price_data else None
            },
            "volatility": {
                "level": volatility_level,
                "value": format_number(volatility_pct, 2)
            },
            "trend": {
                "direction": trend_strength,
                "strength": "strong" if abs(macd_analysis["current_histogram"]["value"]) > 0.01 else "weak"
            },
            "indicators": {
                "RSI": {
                    "current": {
                        "value": format_number(rsi),
                        "trend": rsi_analysis["trend"]
                    },
                    "evolution": format_number(rsi_evolution),
                    "historical": [format_number(v) for v in rsi_values]
                },
                "MACD": macd_analysis,
                "EMA": ema_analysis
            }
        }
        
    def _get_trend_strength(self, rsi: float, macd_hist: float, ema_align: str) -> str:
        """Calculate trend strength based on indicators"""
        strength = 0
        
        # RSI contribution
        if 30 <= rsi <= 70:
            strength += 0
        elif (rsi < 30 and macd_hist > 0) or (rsi > 70 and macd_hist < 0):
            strength += 1
        else:
            strength -= 1
            
        # MACD contribution
        if abs(macd_hist) > 0.01:
            strength += 1 if macd_hist > 0 else -1
            
        # EMA contribution
        if ema_align == "bullish":
            strength += 1
        elif ema_align == "bearish":
            strength -= 1
            
        if strength >= 2:
            return "strong_bullish"
        elif strength == 1:
            return "bullish"
        elif strength == 0:
            return "neutral"
        elif strength == -1:
            return "bearish"
        else:
            return "strong_bearish"
