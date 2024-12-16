import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

class AISTDataFetcher:
    """Simple data fetcher for AI Short Term bot using Polygon API."""
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is not set")
        self.base_url = "https://api.polygon.io/v2"
        
    def _get_headers(self):
        """Get headers for Polygon API requests."""
        return {"Authorization": f"Bearer {self.api_key}"}
    
    def _format_ticker(self, ticker: str) -> str:
        """Format cryptocurrency ticker for Polygon API."""
        ticker = ticker.upper().replace("USD", "").replace("X:", "").replace("-", "")
        return f"X:{ticker}USD"
    
    def get_bars(self, ticker: str, timespan="day", limit=100) -> pd.DataFrame:
        """Get price bars from Polygon API.
        
        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC')
            timespan (str): Time interval ('minute', 'hour', 'day', etc.)
            limit (int): Number of bars to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        formatted_ticker = self._format_ticker(ticker)
        url = f"{self.base_url}/aggs/ticker/{formatted_ticker}/range/1/{timespan}/2020-01-01/{datetime.now().strftime('%Y-%m-%d')}"
        
        try:
            response = requests.get(url, headers=self._get_headers(), params={"limit": limit})
            response.raise_for_status()
            data = response.json()
            
            if "results" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["results"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume", "vwap", "n"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.set_index("timestamp").sort_index()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching bars: {e}")
            return pd.DataFrame()
    
    def get_indicators(self, ticker: str, timespan="day", multiplier=1, limit=10, 
                      rsi_period=14, ema_period=50, sma_period=50,
                      macd_fast=12, macd_slow=26, macd_signal=9):
        """Get technical indicators from Polygon API.
        
        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC')
            timespan (str): Time interval ('minute', 'hour', 'day', etc.)
            multiplier (int): Timespan multiplier (e.g., 5 for 5-minute bars)
            limit (int): Number of data points
            rsi_period (int): RSI period
            ema_period (int): EMA period
            sma_period (int): SMA period
            macd_fast (int): MACD fast period
            macd_slow (int): MACD slow period
            macd_signal (int): MACD signal period
            
        Returns:
            dict: Dictionary containing technical indicators
        """
        formatted_ticker = self._format_ticker(ticker)
        indicators = {}
        
        # Define indicator endpoints and parameters
        indicator_configs = {
            "sma": {"window": sma_period, "series_type": "close"},
            "ema": {"window": ema_period, "series_type": "close"},
            "rsi": {"window": rsi_period, "series_type": "close"},
            "macd": {
                "short_window": macd_fast,
                "long_window": macd_slow,
                "signal_window": macd_signal,
                "series_type": "close"
            }
        }
        
        base_params = {
            "timespan": timespan,
            "multiplier": multiplier,
            "limit": limit,
            "order": "desc",
            "apiKey": self.api_key
        }
        
        for indicator, params in indicator_configs.items():
            url = f"https://api.polygon.io/v1/indicators/{indicator}/{formatted_ticker}"
            try:
                # Combine base parameters with indicator-specific parameters
                request_params = params.copy()
                request_params.update(base_params)
                
                response = requests.get(url, params=request_params)
                response.raise_for_status()
                data = response.json()
                
                if "results" in data:
                    results_df = pd.DataFrame(data["results"]["values"])
                    if not results_df.empty:
                        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"], unit="ms")
                        results_df.set_index("timestamp", inplace=True)
                        results_df.sort_index(inplace=True)
                        indicators[indicator] = results_df
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {indicator}: {e}")
                indicators[indicator] = pd.DataFrame()
        
        return indicators
    
    def get_market_data(self, ticker: str, timespan="day", limit=100):
        """Get both price data and indicators.
        
        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC')
            timespan (str): Time interval ('minute', 'hour', 'day', etc.)
            limit (int): Number of data points
            
        Returns:
            tuple: (price_data DataFrame, indicators dict)
        """
        bars = self.get_bars(ticker, timespan, limit)
        indicators = self.get_indicators(ticker, timespan, limit=limit)
        return bars, indicators
