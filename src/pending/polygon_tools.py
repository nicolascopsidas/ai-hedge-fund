import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


def get_polygon_headers() -> Dict[str, str]:
    """Get headers for Polygon API requests."""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable is not set")
    return {"Authorization": f"Bearer {POLYGON_API_KEY}"}


def format_crypto_ticker(ticker: str) -> str:
    """Format a cryptocurrency ticker for the Polygon API.
    
    Args:
        ticker (str): The cryptocurrency ticker (e.g., 'BTC', 'ETH')
    
    Returns:
        str: Formatted ticker symbol in X:BTCUSD format
    """
    # Remove any existing formatting
    ticker = ticker.upper().replace("USD", "").replace("X:", "").replace("-", "")
    return f"X:{ticker}USD"


def format_date(date_str: str) -> str:
    """Format a date string to Unix timestamp in milliseconds."""
    # Parse the input date string
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Convert to Unix timestamp in milliseconds
    timestamp_ms = int(date.timestamp() * 1000)
    
    return str(timestamp_ms)


def get_prices(ticker: str, start_date: str, end_date: str):
    """Fetch cryptocurrency price data from Polygon.io.
    
    Args:
        ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set")
    
    # Format dates
    formatted_start = format_date(start_date)
    formatted_end = format_date(end_date)
    
    # Format ticker for crypto
    ticker = format_crypto_ticker(ticker)
    
    # Construct API URL
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{formatted_start}/{formatted_end}"
    
    # Make API request with API key in query parameters
    response = requests.get(
        url,
        params={
            "apiKey": POLYGON_API_KEY,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
    )

    if response.status_code != 200:
        raise ValueError(f"Error fetching price data: {response.status_code} - {response.text}")

    data = response.json()
    
    if "results" not in data or not data["results"]:
        raise ValueError(f"No price data available for {ticker} between {start_date} and {end_date}")

    # Convert to DataFrame
    df = pd.DataFrame(data["results"])
    
    # Rename columns to standard names
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap"
    })
    
    # Convert timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    return df


def get_crypto_metrics(ticker: str):
    """Get crypto-specific metrics from Polygon.io
    
    Args:
        ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        dict: Dictionary containing crypto metrics
    """
    formatted_ticker = format_crypto_ticker(ticker)
    
    # Get daily aggregates for the ticker
    url = f"https://api.polygon.io/v2/aggs/ticker/{formatted_ticker}/prev"
    
    response = requests.get(
        url,
        params={"apiKey": POLYGON_API_KEY}
    )
    
    if response.status_code != 200:
        print(f"Error getting crypto metrics: {response.status_code}")
        return get_default_crypto_metrics()
        
    try:
        data = response.json()
        if not data.get("results"):
            print(f"No results found for {formatted_ticker}")
            return get_default_crypto_metrics()
            
        result = data["results"][0]
        
        # Calculate market cap using closing price and a fixed supply for major cryptos
        supply_map = {
            "BTC": 21000000,  # Bitcoin max supply
            "ETH": 120000000,  # Ethereum approximate supply
            "DOGE": 140000000000,  # Dogecoin approximate supply
            "SOL": 550000000,  # Solana approximate supply
        }
        
        base_ticker = ticker.upper().replace("USD", "").replace("X:", "").replace("-", "")
        circulating_supply = supply_map.get(base_ticker, 0)
        market_cap = result.get("c", 0) * circulating_supply
        
        return {
            "market_cap": market_cap,
            "volume_24h": result.get("v", 0) * result.get("c", 0),  # volume in USD
            "vwap_24h": result.get("vw", 0),
            "open_24h": result.get("o", 0),
            "close_24h": result.get("c", 0),
            "high_24h": result.get("h", 0),
            "low_24h": result.get("l", 0),
            "transactions_24h": result.get("n", 0),
            "circulating_supply": circulating_supply,
            "max_supply": supply_map.get(base_ticker, 0),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"Error processing crypto metrics: {e}")
        return get_default_crypto_metrics()


def get_default_crypto_metrics():
    """Return default crypto metrics when data is unavailable."""
    return {
        "market_cap": 0,
        "volume_24h": 0,
        "vwap_24h": 0,
        "open_24h": 0,
        "close_24h": 0,
        "high_24h": 0,
        "low_24h": 0,
        "transactions_24h": 0,
        "circulating_supply": 0,
        "max_supply": 0,
        "last_updated": ""
    }


def get_price_data(ticker: str, start_date: str, end_date: str):
    """Get cryptocurrency price data from Polygon API.
    
    Args:
        ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    return get_prices(ticker, start_date, end_date)


def get_technical_indicators(ticker: str, start_date: str = None):
    """Get technical indicators from Polygon.io for a given crypto ticker
    
    Args:
        ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to today.
    
    Returns:
        dict: Dictionary containing technical indicators
    """
    ticker = format_crypto_ticker(ticker)
    
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")

    # Get MACD
    macd_url = f"https://api.polygon.io/v1/indicators/macd/{ticker}"
    macd_params = {
        "timespan": "day",
        "short_window": 12,
        "long_window": 26,
        "signal_window": 9,
        "series_type": "close",
        "order": "desc",
        "limit": 1,
        "apiKey": POLYGON_API_KEY
    }
    
    # Get RSI
    rsi_url = f"https://api.polygon.io/v1/indicators/rsi/{ticker}"
    rsi_params = {
        "timespan": "day",
        "window": 14,
        "series_type": "close",
        "order": "desc",
        "limit": 1,
        "apiKey": POLYGON_API_KEY
    }
    
    # Get SMA
    sma_url = f"https://api.polygon.io/v1/indicators/sma/{ticker}"
    sma_params = {
        "timespan": "day",
        "window": 50,
        "series_type": "close",
        "order": "desc",
        "limit": 1,
        "apiKey": POLYGON_API_KEY
    }
    
    # Get EMA
    ema_url = f"https://api.polygon.io/v1/indicators/ema/{ticker}"
    ema_params = {
        "timespan": "day",
        "window": 50,
        "series_type": "close",
        "order": "desc",
        "limit": 1,
        "apiKey": POLYGON_API_KEY
    }
    
    # Make API calls
    macd_response = requests.get(macd_url, params=macd_params)
    rsi_response = requests.get(rsi_url, params=rsi_params)
    sma_response = requests.get(sma_url, params=sma_params)
    ema_response = requests.get(ema_url, params=ema_params)
    
    # Extract values
    try:
        macd_data = macd_response.json()["results"]["values"][0]
        macd_value = {
            "macd_line": macd_data.get("value", 0),
            "signal_line": macd_data.get("signal", 0),
            "histogram": macd_data.get("histogram", 0)
        }
    except:
        macd_value = {"macd_line": 0, "signal_line": 0, "histogram": 0}
        
    try:
        rsi_value = rsi_response.json()["results"]["values"][0].get("value", 0)
    except:
        rsi_value = 0
        
    try:
        sma_value = sma_response.json()["results"]["values"][0].get("value", 0)
    except:
        sma_value = 0
        
    try:
        ema_value = ema_response.json()["results"]["values"][0].get("value", 0)
    except:
        ema_value = 0
    
    return {
        "macd": macd_value,
        "rsi": rsi_value,
        "sma": sma_value,
        "ema": ema_value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def calculate_macd(prices_df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices_df (pd.DataFrame): DataFrame with price data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, MACD histogram)
        
    Raises:
        ValueError: If not enough data points for calculation
    """
    min_periods = max(fast_period, slow_period, signal_period)
    if len(prices_df) < min_periods:
        raise ValueError(f"Not enough data points for MACD calculation. Need at least {min_periods}, got {len(prices_df)}")
        
    # Calculate EMAs
    exp1 = prices_df['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = prices_df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line and signal line
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    
    return macd, signal, hist


def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices_df (pd.DataFrame): DataFrame with price data
        period (int): RSI period
        
    Returns:
        pd.Series: RSI values
        
    Raises:
        ValueError: If not enough data points for calculation
    """
    if len(prices_df) < period:
        raise ValueError(f"Not enough data points for RSI calculation. Need at least {period}, got {len(prices_df)}")
        
    # Calculate price changes
    delta = prices_df['close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices_df (pd.DataFrame): DataFrame with price data
        window (int): Bollinger Bands window
        
    Returns:
        tuple: (Upper band, Lower band)
        
    Raises:
        ValueError: If not enough data points for calculation
    """
    if len(prices_df) < window:
        raise ValueError(f"Not enough data points for Bollinger Bands calculation. Need at least {window}, got {len(prices_df)}")
        
    # Calculate SMA
    sma = prices_df['close'].rolling(window=window).mean()
    
    # Calculate standard deviation
    std = prices_df['close'].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    
    return upper_band, lower_band


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        prices_df (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.Series: OBV values
    """
    obv = (prices_df['volume'] *
           (~prices_df['close'].diff().le(0) * 2 - 1)).cumsum()
    
    return obv


def calculate_confidence_level(signals):
    """Calculate confidence level based on the difference between SMAs."""
    # Convert signals to numeric if they're strings
    if isinstance(signals[0], str):
        signal_map = {'buy': 1, 'hold': 0, 'sell': -1}
        signals = [signal_map.get(s, 0) for s in signals]
    
    # Calculate the mean of signals
    mean_signal = sum(signals) / len(signals)
    
    # Scale to confidence level between 0 and 1
    confidence = abs(mean_signal)
    
    return min(confidence, 1.0)  # Cap at 1.0
