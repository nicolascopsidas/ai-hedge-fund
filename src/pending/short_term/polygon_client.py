import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dotenv import load_dotenv

from ..polygon_tools import format_crypto_ticker, format_date

load_dotenv()

class PolygonClient:
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is not set")
        
        self.timeframe_map = {
            "5min": ("minute", 5),
            "15min": ("minute", 15),
            "1hour": ("hour", 1),
            "4hour": ("hour", 4),
            "1day": ("day", 1)
        }
        
        self.base_url = "https://api.polygon.io"
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the Polygon API."""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise ValueError(f"Error making request: {response.status_code} - {response.text}")
            
        return response.json()

    def get_crypto_bars(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get cryptocurrency price bars from Polygon.
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC', 'ETH')
            timeframe (str): Time interval ('5min', '15min', '1hour', '4hour', '1day')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Price data with OHLCV columns
        """
        timespan, multiplier = self.timeframe_map[timeframe]
        formatted_symbol = format_crypto_ticker(symbol)
        formatted_start = format_date(start_date)
        formatted_end = format_date(end_date)
        
        endpoint = f"/v2/aggs/ticker/{formatted_symbol}/range/{multiplier}/{timespan}/{formatted_start}/{formatted_end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        response = self._make_request(endpoint, params)
        
        if "results" not in response or not response["results"]:
            raise ValueError(f"No price data available for {symbol} between {start_date} and {end_date}")
        
        df = pd.DataFrame(response["results"])
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap"
        })
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        return df

    def get_technical_indicators(self, symbol: str, timeframe: str, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get technical indicators from Polygon for a crypto symbol.
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC', 'ETH')
            timeframe (str): Time interval ('5min', '15min', '1hour', '4hour', '1day')
            start_date (str, optional): Start date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Technical indicators data
        """
        if start_date is None:
            # Adjust historical data window based on timeframe
            days_window = {
                "5min": 30,
                "15min": 30,
                "1hour": 45,
                "4hour": 90,  
                "1day": 120
            }.get(timeframe, 30)
            
            start_date = (datetime.now() - timedelta(days=days_window)).strftime("%Y-%m-%d")
            
        # Get price data with enough history for indicators
        df = self.get_crypto_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        if df.empty:
            return pd.DataFrame()
            
        # Ensure we have enough data points
        min_points = 30
        if len(df) < min_points:
            print(f"Warning: Not enough data points for {timeframe} timeframe. Need at least {min_points}, got {len(df)}")
            return pd.DataFrame()
        
        # Calculate indicators
        results = {}
        
        # Adjust window sizes based on timeframe
        windows = {
            "5min": {"sma": 20, "rsi": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
            "15min": {"sma": 20, "rsi": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
            "1hour": {"sma": 20, "rsi": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9},
            "4hour": {"sma": 10, "rsi": 7, "macd_fast": 8, "macd_slow": 17, "macd_signal": 9},
            "1day": {"sma": 10, "rsi": 7, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
        }.get(timeframe, {"sma": 20, "rsi": 14, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
        
        # SMA
        results['sma_values'] = df['close'].rolling(window=windows["sma"], min_periods=1).mean()
        
        # EMA
        results['ema_values'] = df['close'].ewm(span=windows["sma"], adjust=False, min_periods=1).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=windows["rsi"], min_periods=1).mean()
        avg_loss = loss.rolling(window=windows["rsi"], min_periods=1).mean()
        rs = avg_gain / avg_loss
        results['rsi_values'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=windows["macd_fast"], adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=windows["macd_slow"], adjust=False, min_periods=1).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=windows["macd_signal"], adjust=False, min_periods=1).mean()
        results['macd_values'] = macd - signal
        
        # Convert to DataFrame
        indicators_df = pd.DataFrame(results, index=df.index)
        
        # Forward fill and backward fill missing values
        indicators_df = indicators_df.ffill().bfill()
        
        return indicators_df

    def get_order_book(self, symbol: str) -> Dict:
        """Get real-time order book data for a crypto symbol.
        
        Args:
            symbol (str): Crypto symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dict: Order book data with bids and asks
        """
        formatted_symbol = format_crypto_ticker(symbol)
        endpoint = f"/v2/snapshot/l2/{formatted_symbol}"
        
        try:
            response = self._make_request(endpoint)
            if response["status"] == "OK" and "data" in response:
                return response["data"]
        except Exception as e:
            print(f"Error fetching order book: {e}")
            
        return {"bids": [], "asks": []}
