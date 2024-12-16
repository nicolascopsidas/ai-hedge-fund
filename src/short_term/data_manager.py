from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from src.short_term.models import TimeFrame
from src.short_term.polygon_client import PolygonClient


class MultiTimeframeData:
    def __init__(self, ticker: str, cache_duration: Optional[timedelta] = None):
        """Initialize the multi-timeframe data manager.
        
        Args:
            ticker (str): The cryptocurrency ticker (e.g., 'BTC')
            cache_duration (timedelta, optional): How long to cache data before refreshing
        """
        self.ticker = ticker
        self.polygon = PolygonClient()
        self.cache_duration = cache_duration or timedelta(minutes=1)
        self.timeframes: Dict[TimeFrame, pd.DataFrame] = {
            tf: pd.DataFrame() for tf in TimeFrame
        }
        self.last_update: Dict[TimeFrame, datetime] = {
            tf: datetime.min for tf in TimeFrame
        }
        self.indicators_cache: Dict[TimeFrame, pd.DataFrame] = {
            tf: pd.DataFrame() for tf in TimeFrame
        }
        self.data_cache: Dict[str, Dict] = {}  # Cache for get_data results
    
    def _get_cache_key(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime) -> str:
        """Generate a unique cache key for data requests."""
        return f"{symbol}_{timeframe.value}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    def _should_update(self, timeframe: TimeFrame) -> bool:
        """Check if the data for a timeframe needs updating."""
        now = datetime.now()
        last_update = self.last_update[timeframe]
        
        # Define update frequencies based on timeframe
        update_intervals = {
            TimeFrame.M5: timedelta(minutes=1),
            TimeFrame.M15: timedelta(minutes=5),
            TimeFrame.H1: timedelta(minutes=15),
            TimeFrame.H4: timedelta(hours=1),
            TimeFrame.D1: timedelta(hours=4),
        }
        
        # Check if enough time has passed since last update
        if now - last_update > update_intervals[timeframe]:
            # For intraday timeframes, only update during market hours
            if timeframe in [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]:
                # Crypto markets are 24/7, so always update
                return True
            # For larger timeframes, update less frequently
            return True
        return False
    
    def update_timeframe(self, timeframe: TimeFrame) -> None:
        """Update data for a specific timeframe using Polygon API."""
        if not self._should_update(timeframe):
            return
            
        end_date = datetime.now()
        start_date = end_date - TimeFrame.get_lookback(timeframe)
        
        try:
            # Get price data from Polygon
            df = self.polygon.get_crypto_bars(
                self.ticker,
                timeframe.value,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if not df.empty:
                self.timeframes[timeframe] = df
                self.last_update[timeframe] = datetime.now()
                
                # Update technical indicators
                self._update_indicators(timeframe)
                
        except Exception as e:
            print(f"Error updating {timeframe.value} data: {e}")
    
    def _update_indicators(self, timeframe: TimeFrame) -> None:
        """Update technical indicators for a timeframe."""
        try:
            df = self.polygon.get_technical_indicators(
                self.ticker,
                timeframe.value,
                start_date=(datetime.now() - TimeFrame.get_lookback(timeframe)).strftime("%Y-%m-%d")
            )
            
            if not df.empty:
                self.indicators_cache[timeframe] = df
                
        except Exception as e:
            print(f"Error updating indicators for {timeframe.value}: {e}")
    
    def update_all(self) -> None:
        """Update all timeframes."""
        for timeframe in TimeFrame:
            self.update_timeframe(timeframe)
    
    def get_data(self, symbol: str, timeframe: TimeFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get price data and indicators for a symbol at a specific timeframe.
        
        Args:
            symbol (str): The cryptocurrency symbol (e.g., 'BTC')
            timeframe (TimeFrame): The timeframe to fetch data for
            start_date (datetime): Start date for data fetching
            end_date (datetime): End date for data fetching
            
        Returns:
            pd.DataFrame: Combined price and indicator data
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(symbol, timeframe, start_date, end_date)
            if cache_key in self.data_cache:
                cache_entry = self.data_cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                    return cache_entry['data']
            
            # Get price data from Polygon
            df = self.polygon.get_crypto_bars(
                symbol=symbol,
                timeframe=timeframe.value,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            
            if df.empty:
                print(f"No data available for {symbol} at {timeframe}")
                return pd.DataFrame()
            
            # Get technical indicators with enough history for calculations
            indicator_start = start_date - TimeFrame.get_lookback(timeframe)
            indicators_df = self.polygon.get_technical_indicators(
                symbol=symbol,
                timeframe=timeframe.value,
                start_date=indicator_start.strftime("%Y-%m-%d")
            )
            
            if not indicators_df.empty:
                # Merge indicators with price data
                df = df.join(indicators_df, how='left')
                
                # Forward fill and backward fill missing values
                df = df.ffill().bfill()
            
            # Cache the result
            self.data_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, timeframe: TimeFrame) -> float:
        """Get the latest price for a specific timeframe."""
        df = self.get_data(self.ticker, timeframe, datetime.now() - timedelta(days=1), datetime.now())
        if df.empty:
            raise ValueError(f"No data available for {timeframe.value}")
        return df['close'].iloc[-1]
    
    def get_latest_timestamp(self, timeframe: TimeFrame) -> datetime:
        """Get the timestamp of the latest data point."""
        df = self.get_data(self.ticker, timeframe, datetime.now() - timedelta(days=1), datetime.now())
        if df.empty:
            raise ValueError(f"No data available for {timeframe.value}")
        return df.index[-1]
    
    def get_order_book(self) -> Dict:
        """Get current order book data."""
        return self.polygon.get_order_book(self.ticker)
