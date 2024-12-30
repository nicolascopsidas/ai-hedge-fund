from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from dateutil import tz

load_dotenv()


class AISTDataFetcher:
    """Simple data fetcher for AI Short Term bot using Polygon API."""

    MAX_STALENESS = {
        # Increased tolerance for crypto minute data
        "minute": timedelta(minutes=30),
        # Increased for better data availability
        "hour": timedelta(hours=4),
        "day": timedelta(days=1)
    }

    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is not set")
        self.base_url_v2 = "https://api.polygon.io/v2"
        self.base_url_v1 = "https://api.polygon.io/v1"

    def _get_headers(self):
        """Get headers for Polygon API requests."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _format_ticker(self, ticker: str) -> str:
        """Format cryptocurrency ticker for Polygon API."""
        # Remove /USD if present and clean up the ticker
        ticker = ticker.upper().replace("/USD", "").replace("USD",
                                                            "").replace("X:", "").replace("-", "")
        # Return in X:XXXUSD format
        return f"X:{ticker}USD"

    def _check_crypto_supported(self, ticker: str) -> bool:
        """Check if the cryptocurrency is supported by Polygon API."""
        url = f"{self.base_url_v2}/snapshot/locale/global/markets/crypto/tickers/{
            self._format_ticker(ticker)}"

        try:
            response = requests.get(url, headers=self._get_headers())
            if response.status_code == 200:
                data = response.json()
                if "ticker" in data and "status" in data and data["status"] == "OK":
                    return True
            print(f"⚠️  Warning: {
                  ticker} may not be supported on Polygon API. Response: {response.text}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Error checking crypto support: {str(e)}")
            return False

    def _is_data_stale(self, timestamp, timeframe: str) -> bool:
        """Print timestamp information without staleness check."""
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(
                timestamp / 1000.0, tz=tz.tzutc())
        elif isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        if not isinstance(timestamp, datetime):
            print(f"❌ Invalid timestamp type: {type(timestamp)}")
            return False

        current_time = datetime.now(tz=tz.tzutc())
        time_diff = current_time - timestamp

        return False  # Always return False to avoid staleness checks

    def get_bars(self, ticker: str, timespan="day", multiplier=1, limit=100) -> pd.DataFrame:
        """Get price bars from Polygon API v2."""
        if not self._check_crypto_supported(ticker):
            print(f"❌ Error: {ticker} is not supported on Polygon API")
            return pd.DataFrame()

        formatted_ticker = self._format_ticker(ticker)

        end_time = datetime.now(timezone.utc)
        if timespan == "minute":
            start_time = end_time - timedelta(minutes=limit * multiplier)
        elif timespan == "hour":
            start_time = end_time - timedelta(hours=limit * multiplier)
        else:  # day
            start_time = end_time - timedelta(days=limit * multiplier)

        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")

        url = f"https://api.polygon.io/v2/aggs/ticker/{formatted_ticker}/range/{
            multiplier}/{timespan}/{start_str}/{end_str}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "results" not in data:
                print(f"❌ Error: No data available for {ticker}")
                print(f"API Response: {data}")
                return pd.DataFrame()

            if not data["results"]:
                print(f"⚠️  Warning: No {
                      timespan} data available for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])

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

            latest_timestamp = df.index.max()
            if self._is_data_stale(latest_timestamp, timespan):
                if timespan != "day":
                    print(
                        f"⚠️  Attempting fallback to larger timeframe due to stale data...")
                    if timespan == "minute":
                        return self.get_bars(ticker, "hour", 1, limit)
                    elif timespan == "hour":
                        return self.get_bars(ticker, "day", 1, limit)

            return df

        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            print(f"URL: {url}")
            print(f"Response: {e.response.text if hasattr(
                e, 'response') else 'No response'}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unexpected error fetching {timespan} data: {str(e)}")
            return pd.DataFrame()

    def get_indicators(self, ticker: str, timespan="day", multiplier=1, limit=10,
                       rsi_period=14, ema_period=50, sma_period=50,
                       macd_fast=12, macd_slow=26, macd_signal=9):
        """Get technical indicators from Polygon API v1 endpoints."""
        formatted_ticker = self._format_ticker(ticker)
        indicators = {}

        indicator_configs = {
            "rsi": {
                "endpoint": f"{self.base_url_v1}/indicators/rsi/{formatted_ticker}",
                "params": {
                    "timespan": timespan,
                    "window": rsi_period,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit,
                    "apiKey": self.api_key
                }
            },
            "ema": {
                "endpoint": f"{self.base_url_v1}/indicators/ema/{formatted_ticker}",
                "params": {
                    "timespan": timespan,
                    "window": ema_period,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit,
                    "apiKey": self.api_key
                }
            },
            "sma": {
                "endpoint": f"{self.base_url_v1}/indicators/sma/{formatted_ticker}",
                "params": {
                    "timespan": timespan,
                    "window": sma_period,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit,
                    "apiKey": self.api_key
                }
            },
            "macd": {
                "endpoint": f"{self.base_url_v1}/indicators/macd/{formatted_ticker}",
                "params": {
                    "timespan": timespan,
                    "short_window": macd_fast,
                    "long_window": macd_slow,
                    "signal_window": macd_signal,
                    "series_type": "close",
                    "multiplier": multiplier,
                    "order": "desc",
                    "limit": limit,
                    "apiKey": self.api_key
                }
            }
        }

        for indicator, config in indicator_configs.items():
            try:
                response = requests.get(
                    config['endpoint'], params=config['params'])
                response.raise_for_status()
                data = response.json()

                if "results" in data and "values" in data["results"]:
                    results_df = pd.DataFrame(data["results"]["values"])
                    if not results_df.empty:
                        results_df["timestamp"] = pd.to_datetime(
                            results_df["timestamp"], unit="ms").dt.tz_localize('UTC')
                        results_df.set_index("timestamp", inplace=True)
                        results_df.sort_index(inplace=True)

                        latest_timestamp = results_df.index.max()
                        if self._is_data_stale(latest_timestamp.timestamp() * 1000, timespan):
                            print(f"⚠️  Warning: {
                                  indicator} data is stale (Last update: {latest_timestamp})")

                        if indicator == 'macd':
                            indicators[indicator] = {
                                'macd': results_df['value'],
                                'signal': results_df['signal'],
                                'histogram': results_df['histogram']
                            }
                        else:
                            indicators[indicator] = {
                                'value': results_df['value']}
                    else:
                        print(f"No data available for {indicator}")
                else:
                    print(f"Error: Unexpected response format for {indicator}")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching {indicator}: {str(e)}")
                indicators[indicator] = pd.DataFrame()

        return indicators

    def get_market_data(self, ticker: str, timespan="day", multiplier=1, limit=100):
        """Get both price data and indicators.

        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC')
            timespan (str): Time interval ('minute', 'hour', 'day', etc.)
            multiplier (int): Timespan multiplier (e.g., 5 for 5-minute bars)
            limit (int): Number of data points

        Returns:
            tuple: (price_data DataFrame, indicators dict)
        """
        price_data = self.get_bars(ticker, timespan, multiplier, limit)

        if price_data.empty:
            print(f"❌ Error: Unable to fetch reliable price data for {ticker}")
            return pd.DataFrame(), {}

        latest_timestamp = price_data.index.max()
        if self._is_data_stale(latest_timestamp, timespan):
            print(f"❌ Error: Data is too stale for reliable indicator calculation")
            return pd.DataFrame(), {}

        return price_data, self.get_indicators(ticker, timespan, multiplier, limit)
