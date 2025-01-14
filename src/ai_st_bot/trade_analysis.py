from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import requests  # Added to fetch data from CryptoCompare
import sys
from typing import Dict, List, Tuple, Optional

from ai_st_data import AISTDataFetcher

class TimeframeAnalyzer:
    """Analyze market data across multiple timeframes and integrate fundamental data."""
    
    def __init__(self, crypto_compare_api_key: str):
        self.fetcher = AISTDataFetcher()
        self.crypto_compare_api_key = crypto_compare_api_key

    def _fetch_news(self, ticker: str) -> List[Dict]:
        """Fetch latest news related to the ticker from CryptoCompare."""
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={ticker}&api_key={self.crypto_compare_api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            news_data = response.json()
            return news_data.get("Data", [])
        except requests.RequestException as e:
            print(f"Error fetching news data: {e}")
            return []

    def _calculate_volatility(self, high: float, low: float) -> Tuple[str, float]:
        """Calculate volatility level and percentage."""
        volatility = (high - low) / low * 100

        if volatility < 0.5:
            return "low", volatility
        elif volatility < 2.0:
            return "medium", volatility
        else:
            return "high", volatility

    # Other indicator methods here (unchanged)...

    def analyze_timeframe(self, ticker: str, timeframe: str) -> dict:
        """Analyze a specific timeframe and include news."""
        config = {
            "5m": {"timespan": "minute", "multiplier": 5},
            "15m": {"timespan": "minute", "multiplier": 15},
            "1h": {"timespan": "hour", "multiplier": 1},
            "4h": {"timespan": "hour", "multiplier": 4},
            "1d": {"timespan": "day", "multiplier": 1}
        }[timeframe]

        price_data, indicators = self.fetcher.get_market_data(ticker, config['timespan'], config['multiplier'])

        if price_data.empty or not indicators:
            return None

        latest_price = price_data['close'].iloc[-1]
        high = price_data['high'].max()
        low = price_data['low'].min()

        volatility_level, volatility_pct = self._calculate_volatility(high, low)

        # Fetch news
        news = self._fetch_news(ticker)

        return {
            "timeframe": timeframe,
            "price": {
                "current": {
                    "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                    "value": round(float(latest_price), 6),
                    "description": "Prix actuel de l'actif"
                }
            },
            "volatility": volatility_level,
            "news": news  # Adding news to the analysis
        }

    def analyze_all(self, symbol: str) -> dict:
        """Analyze all timeframes and generate complete analysis."""
        timeframes = ["5m", "15m", "1h", "4h", "1d"]  
        timeframes_data = {}

        # Analyze each timeframe
        for timeframe in timeframes:
            timeframes_data[timeframe] = self.analyze_timeframe(symbol, timeframe)

        # Skip if we don't have enough data
        if not any(timeframes_data.values()):
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "error": "Insufficient data for analysis"
            }

        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "timeframes": timeframes_data
        }

def main():
    """Main function to run the analyzer."""
    if len(sys.argv) != 2:
        print("Usage: python timeframe_analyzer.py <symbol>")
        sys.exit(1)

    symbol = sys.argv[1]
    # Add /USD only if not already present
    if not symbol.endswith("/USD"):
        symbol = f"{symbol}/USD"

    api_key = "4dbe373fb75eede45f6213059fa46abd26e704e6fd83278b9135a193918c8edb"
    analyzer = TimeframeAnalyzer(api_key)
    analysis = analyzer.analyze_all(symbol)

    # Pretty print the analysis
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
