"""Multi-timeframe confluence analysis module"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import argparse
import json
import sys

from src.ai_trading_bot.data.fetcher import MarketDataFetcher
from src.ai_trading_bot.analysis.timeframe import TimeframeAnalyzer

class ConfluenceAnalyzer:
    """Analyzes market data across multiple timeframes for confluence"""
    
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.timeframe_analyzer = TimeframeAnalyzer()
        
    def analyze_timeframes(
        self,
        symbol: str,
        timeframes: Optional[List[Dict[str, any]]] = None
    ) -> Dict[str, any]:
        """Analyze market data across multiple timeframes"""
        if timeframes is None:
            timeframes = [
                {"timespan": "minute", "multiplier": 5},
                {"timespan": "minute", "multiplier": 15},
                {"timespan": "hour", "multiplier": 1},
                {"timespan": "hour", "multiplier": 4},
                {"timespan": "day", "multiplier": 1}
            ]
            
        timeframes_data = {}
        
        # Analyze each timeframe
        for tf in timeframes:
            timeframe_key = f"{tf['multiplier']}{tf['timespan']}"
            
            price_data = self.fetcher.get_price_data(
                symbol,
                timespan=tf["timespan"],
                multiplier=tf["multiplier"]
            )
            
            if price_data.empty:
                timeframes_data[timeframe_key] = None
                continue
                
            indicators = self.fetcher.get_indicators(
                symbol,
                timespan=tf["timespan"],
                multiplier=tf["multiplier"]
            )
            
            timeframes_data[timeframe_key] = self.timeframe_analyzer.analyze_timeframe(
                price_data,
                indicators
            )
            
        # If no data available for any timeframe, return error
        if not any(timeframes_data.values()):
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "error": "Insufficient data for analysis"
            }
            
        # Identify confluence zones
        confluence_zones = self._identify_confluence_zones(timeframes_data)
        
        # Generate trade suggestion
        trade_suggestion = self._generate_trade_suggestion(
            timeframes_data,
            confluence_zones
        )
        
        # Calculate overall market bias and confidence
        bias, confidence = self._calculate_market_bias(timeframes_data)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "timeframes": timeframes_data,
            "summary": {
                "bias": bias,
                "confidence_level": confidence,
                "confluence_zones": confluence_zones,
                "trade_suggestion": trade_suggestion
            }
        }
        
    def _identify_confluence_zones(
        self,
        timeframes_data: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """Identify high and low confidence confluence zones"""
        high_confidence = []
        low_confidence = []
        
        for timeframe, data in timeframes_data.items():
            if not data:
                continue
                
            # Check for strong trends with multiple aligned indicators
            if (data["trend"]["strength"] == "strong" and
                len(data["indicators"]["EMA"]["aligned_indicators"]) >= 2):
                high_confidence.append(f"{timeframe}_{data['trend']['direction']}")
            elif data["trend"]["strength"] != "weak":
                low_confidence.append(f"{timeframe}_{data['trend']['direction']}")
                
        return {
            "high_confidence": high_confidence,
            "low_confidence": low_confidence
        }
        
    def _generate_trade_suggestion(
        self,
        timeframes_data: Dict[str, Dict],
        confluence_zones: Dict[str, List[str]]
    ) -> Dict[str, any]:
        """Generate trade suggestion based on analysis"""
        if not confluence_zones["high_confidence"] and not confluence_zones["low_confidence"]:
            return {
                "action": "wait",
                "reason": "No clear confluence zones identified"
            }
            
        # Count bullish and bearish signals in high confidence zones
        high_conf_bullish = len([z for z in confluence_zones["high_confidence"] if "bullish" in z])
        high_conf_bearish = len([z for z in confluence_zones["high_confidence"] if "bearish" in z])
        
        if high_conf_bullish > high_conf_bearish:
            return {
                "action": "buy",
                "reason": "Strong bullish confluence across multiple timeframes"
            }
        elif high_conf_bearish > high_conf_bullish:
            return {
                "action": "sell",
                "reason": "Strong bearish confluence across multiple timeframes"
            }
        else:
            return {
                "action": "wait",
                "reason": "Mixed signals across timeframes"
            }
            
    def _calculate_market_bias(
        self,
        timeframes_data: Dict[str, Dict]
    ) -> Tuple[str, str]:
        """Calculate overall market bias and confidence level"""
        bullish_count = 0
        bearish_count = 0
        strong_trends = 0
        total_valid_timeframes = 0
        
        for data in timeframes_data.values():
            if not data:
                continue
                
            total_valid_timeframes += 1
            
            if data["trend"]["direction"] == "bullish":
                bullish_count += 1
            elif data["trend"]["direction"] == "bearish":
                bearish_count += 1
                
            if data["trend"]["strength"] == "strong":
                strong_trends += 1
                
        # Determine bias
        if bullish_count > bearish_count:
            bias = "bullish"
        elif bearish_count > bullish_count:
            bias = "bearish"
        else:
            bias = "neutral"
            
        # Determine confidence
        confidence_ratio = strong_trends / total_valid_timeframes if total_valid_timeframes > 0 else 0
        
        if confidence_ratio >= 0.6:
            confidence = "high"
        elif confidence_ratio >= 0.3:
            confidence = "medium"
        else:
            confidence = "low"
            
        return bias, confidence

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Trading Bot - Technical Analysis")
    
    parser.add_argument(
        "symbol",
        type=str,
        help="Trading symbol (e.g., BTC, ETH, RUNE)"
    )
    
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        help="List of timeframes to analyze (e.g., '5minute' '1hour' '1day')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (pretty or json)"
    )
    
    return parser.parse_args()

def parse_timeframe(timeframe: str) -> Optional[Dict[str, any]]:
    """Parse timeframe string into components"""
    if not timeframe:
        return None
        
    # Extract multiplier and timespan
    for span in ["minute", "hour", "day"]:
        if span in timeframe:
            multiplier = timeframe.replace(span, "")
            try:
                return {
                    "timespan": span,
                    "multiplier": int(multiplier)
                }
            except ValueError:
                return None
                
    return None

def format_pretty_output(analysis: Dict[str, any]) -> None:
    """Format analysis results in a human-readable way"""
    print("\n=== AI Trading Bot Analysis ===")
    print(f"Symbol: {analysis['symbol']}")
    print(f"Timestamp: {analysis['timestamp']}")
    
    if "error" in analysis:
        print(f"\nError: {analysis['error']}")
        return
        
    print("\n=== Market Summary ===")
    summary = analysis["summary"]
    print(f"Market Bias: {summary['bias'].upper()}")
    print(f"Confidence Level: {summary['confidence_level'].upper()}")
    
    print("\n=== Confluence Zones ===")
    zones = summary["confluence_zones"]
    if zones["high_confidence"]:
        print("\nHigh Confidence Zones:")
        for zone in zones["high_confidence"]:
            print(f"- {zone}")
    if zones["low_confidence"]:
        print("\nLow Confidence Zones:")
        for zone in zones["low_confidence"]:
            print(f"- {zone}")
            
    print("\n=== Trade Suggestion ===")
    suggestion = summary["trade_suggestion"]
    print(f"Action: {suggestion['action'].upper()}")
    print(f"Reason: {suggestion['reason']}")
    
    print("\n=== Timeframe Analysis ===")
    for timeframe, data in analysis["timeframes"].items():
        if not data:
            continue
            
        print(f"\nTimeframe: {timeframe}")
        print(f"Current Price: {data['price']['current']['value']}")
        print(f"Volume: {data['volume']['current']} ({data['volume']['trend']})")
        if data['volume']['vwap']:
            print(f"VWAP: {data['volume']['vwap']}")
            
        print(f"Volatility: {data['volatility']['level'].upper()} ({data['volatility']['value']}%)")
        print(f"Trend: {data['trend']['direction'].upper()} ({data['trend']['strength']})")
        
        # Show key indicators
        rsi = data["indicators"]["RSI"]["current"]["value"]
        print(f"RSI: {rsi}")
        
        macd = data["indicators"]["MACD"]
        print(f"MACD: {macd['trend'].upper()} ({macd['strength']})")
        
        ema = data["indicators"]["EMA"]
        print(f"EMA Trend: {ema['trend'].upper()}")
        if ema["aligned_indicators"]:
            print("Aligned Indicators:", ", ".join(ema["aligned_indicators"]))
            
    print("\n" + "="*30 + "\n")

def main() -> None:
    """Main entry point"""
    args = parse_args()
    
    # Add /USD only if not already present
    symbol = args.symbol
    if not symbol.endswith("/USD"):
        symbol = f"{symbol}/USD"
    
    # Parse timeframes if provided
    timeframes = None
    if args.timeframes:
        timeframes = []
        for tf in args.timeframes:
            parsed = parse_timeframe(tf)
            if parsed:
                timeframes.append(parsed)
                
    # Run analysis
    analyzer = ConfluenceAnalyzer()
    analysis = analyzer.analyze_timeframes(symbol, timeframes)
    
    # Output results
    if args.output == "json":
        print(json.dumps(analysis, indent=2))
    else:
        format_pretty_output(analysis)

if __name__ == "__main__":
    main()
