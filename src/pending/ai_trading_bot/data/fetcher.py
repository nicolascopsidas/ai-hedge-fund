"""Data fetching module for cryptocurrency market data"""

from datetime import datetime, timedelta, timezone
import os
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.ai_trading_bot.utils.config import (
    POLYGON_BASE_URL_V1,
    POLYGON_BASE_URL_V2,
    MAX_STALENESS,
    INDICATOR_SETTINGS
)
from src.ai_trading_bot.utils.formatting import format_timestamp
from src.ai_trading_bot.data.models import PriceData, IndicatorValue

load_dotenv()

class MarketDataFetcher:
    """Fetches market data from Polygon API with caching and staleness checks."""
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
            
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the Polygon API"""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch data from Polygon API: {str(e)}")
            
    def _format_ticker(self, ticker: str) -> str:
        """Format cryptocurrency ticker for Polygon API."""
        # Remove /USD if present and clean up the ticker
        ticker = ticker.upper().replace("/USD", "").replace("USD", "").replace("X:", "").replace("-", "")
        # Return in X:XXXUSD format
        return f"X:{ticker}USD"
            
    def get_price_data(
        self,
        symbol: str,
        timespan: str = "minute",
        multiplier: int = 1,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 5000
    ) -> pd.DataFrame:
        """Fetch price data from Polygon API"""
        if to_date is None:
            to_date = datetime.now(timezone.utc)
        if from_date is None:
            from_date = to_date - MAX_STALENESS[timespan]
            
        # Format ticker for Polygon API
        symbol = self._format_ticker(symbol)
            
        endpoint = f"{POLYGON_BASE_URL_V2}/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
        
        response = self._make_request(endpoint, {"limit": limit})
        
        if "results" not in response:
            print(f"❌ Error: No data available for {symbol}")
            print(f"API Response: {response}")
            return pd.DataFrame()

        if not response["results"]:
            print(f"⚠️  Warning: No {timespan} data available for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(response["results"])

        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap"
        })

        df = df.drop(["t", "n"], axis=1, errors="ignore")
        df = df.set_index("timestamp").sort_index().tail(limit)

        return df
        
    def get_indicators(
        self,
        symbol: str,
        timespan: str = "minute",
        multiplier: int = 1,
        rsi_period: int = 14,
        ema_period: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        limit: int = 5000
    ) -> Dict[str, pd.DataFrame]:
        """Get technical indicators from Polygon API"""
        # Format ticker for Polygon API
        symbol = self._format_ticker(symbol)
        indicators = {}

        indicator_configs = {
            "rsi": {
                "endpoint": f"{POLYGON_BASE_URL_V1}/indicators/rsi/{symbol}",
                "params": {
                    "timespan": timespan,
                    "window": rsi_period,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit
                }
            },
            "ema": {
                "endpoint": f"{POLYGON_BASE_URL_V1}/indicators/ema/{symbol}",
                "params": {
                    "timespan": timespan,
                    "window": ema_period,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit
                }
            },
            "macd": {
                "endpoint": f"{POLYGON_BASE_URL_V1}/indicators/macd/{symbol}",
                "params": {
                    "timespan": timespan,
                    "short_window": macd_fast,
                    "long_window": macd_slow,
                    "signal_window": macd_signal,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit
                }
            }
        }

        for name, config in indicator_configs.items():
            try:
                response = self._make_request(config["endpoint"], config["params"])
                
                if "results" not in response or "values" not in response["results"]:
                    print(f"❌ Error: No {name} data available for {symbol}")
                    print(f"API Response: {response}")
                    continue

                results_df = pd.DataFrame(response["results"]["values"])
                if results_df.empty:
                    print(f"⚠️  Warning: No {name} data available for {symbol}")
                    continue

                results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], unit="ms", utc=True)
                results_df = results_df.set_index("timestamp").sort_index()

                if name == "macd":
                    indicators[name] = {
                        "macd": results_df["value"],
                        "signal": results_df["signal"],
                        "histogram": results_df["histogram"]
                    }
                else:
                    indicators[name] = {
                        "value": results_df["value"]
                    }

            except Exception as e:
                print(f"❌ Error fetching {name} data: {str(e)}")
                continue

        return indicators
