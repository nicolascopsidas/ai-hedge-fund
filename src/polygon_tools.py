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
    """Format a cryptocurrency ticker for the Polygon API."""
    # Remove any existing X: prefix
    ticker = ticker.replace("X:", "")
    
    # Remove USD suffix if present
    ticker = ticker.replace("USD", "")
    
    # Add X: prefix and USD suffix
    return f"X:{ticker}USD"


def format_date(date_str: str) -> str:
    """Format a date string to Unix timestamp in milliseconds."""
    # Parse the input date string
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Convert to Unix timestamp in milliseconds
    timestamp_ms = int(date.timestamp() * 1000)
    
    return str(timestamp_ms)


def get_prices(ticker: str, start_date: str, end_date: str, asset_type: str = 'crypto'):
    """Fetch price data from Polygon.io.
    
    Args:
        ticker (str): The ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        asset_type (str, optional): Type of asset ('crypto' or 'stock'). Defaults to 'crypto'.
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set")
    
    # Format dates
    formatted_start = format_date(start_date)
    formatted_end = format_date(end_date)
    
    # Format ticker for crypto if needed
    if asset_type.lower() == 'crypto':
        ticker = format_crypto_ticker(ticker)
    
    # Construct API URL
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{formatted_start}/{formatted_end}"
    
    # Make API request with API key in query parameters (as per Polygon.io docs)
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
    
    # Set timestamp as index
    df.set_index("timestamp", inplace=True)
    
    return df


def get_financial_metrics(ticker: str, report_period: str, period='ttm', limit=1, asset_type='stock'):
    """Fetch financial metrics from Polygon.io.
    
    For crypto assets, most traditional financial metrics don't apply,
    so we'll return market-specific metrics instead.
    """
    if asset_type.lower() == 'crypto':
        return get_crypto_metrics(ticker, report_period)
    
    headers = get_polygon_headers()
    url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=quarterly&limit={limit}"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error fetching financial metrics: {response.status_code} - {response.text}")
        
    data = response.json()
    results = data.get("results", [])

    if not results:
        raise ValueError("No financial metrics found")
        
    metrics = []
    for result in results:
        financials = result.get("financials", {})
        income_stmt = financials.get("income_statement", {})
        balance_sheet = financials.get("balance_sheet", {})
        total_revenue = income_stmt.get("revenues", {}).get("value", 0)
        net_income = income_stmt.get("net_income_loss", {}).get("value", 0)
        total_assets = balance_sheet.get("assets", {}).get("value", 0)
        total_equity = balance_sheet.get("equity", {}).get("value", 0)

        metrics.append({
            "return_on_equity": net_income / total_equity if total_equity else 0.0,
            "net_margin": net_income / total_revenue if total_revenue else 0.0,
            "operating_margin": income_stmt.get("operating_income_loss", {}).get("value", 0) / total_revenue if total_revenue else 0.0,
            "revenue_growth": 0.0,  # Need historical data to calculate growth
            "earnings_growth": 0.0,  # Need historical data to calculate growth
            "book_value_growth": 0.0,  # Need historical data to calculate growth
            "current_ratio": balance_sheet.get("current_assets", {}).get("value", 0) / balance_sheet.get("current_liabilities", {}).get("value", 1),
            "debt_to_equity": balance_sheet.get("liabilities", {}).get("value", 0) / total_equity if total_equity else 0.0,
            "free_cash_flow_per_share": 0.0,  # Need cash flow statement data
            "earnings_per_share": net_income / financials.get("shares", 1),
            "price_to_earnings_ratio": 0.0,  # Need current price data
            "price_to_book_ratio": 0.0,  # Need current price data
            "price_to_sales_ratio": 0.0  # Need current price data
        })

    return metrics


def get_crypto_metrics(ticker: str, report_period: str):
    """Get crypto-specific metrics from Polygon.io"""
    headers = get_polygon_headers()
    ticker = format_crypto_ticker(ticker)
    
    # Get 24h stats for the crypto
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error fetching crypto metrics: {response.status_code} - {response.text}")
        
    data = response.json()
    result = data.get("results", [{}])[0]
    
    return [{
        "24h_volume": result.get('v', 0),
        "24h_vwap": result.get('vw', 0),
        "24h_open": result.get('o', 0),
        "24h_close": result.get('c', 0),
        "24h_high": result.get('h', 0),
        "24h_low": result.get('l', 0),
        "24h_transactions": result.get('n', 0),
        # Traditional metrics set to 0 for crypto
        "return_on_equity": 0.0,
        "net_margin": 0.0,
        "operating_margin": 0.0,
        "revenue_growth": 0.0,
        "earnings_growth": 0.0,
        "book_value_growth": 0.0,
        "current_ratio": 0.0,
        "debt_to_equity": 0.0,
        "free_cash_flow_per_share": 0.0,
        "earnings_per_share": 0.0,
        "price_to_earnings_ratio": 0.0,
        "price_to_book_ratio": 0.0,
        "price_to_sales_ratio": 0.0
    }]


def get_default_metrics():
    """Return default metrics when data is unavailable."""
    return {
        "return_on_equity": 0.0,
        "net_margin": 0.0,
        "operating_margin": 0.0,
        "revenue_growth": 0.0,
        "earnings_growth": 0.0,
        "book_value_growth": 0.0,
        "current_ratio": 0.0,
        "debt_to_equity": 0.0,
        "free_cash_flow_per_share": 0.0,
        "earnings_per_share": 0.0,
        "price_to_earnings_ratio": 0.0,
        "price_to_book_ratio": 0.0,
        "price_to_sales_ratio": 0.0
    }


def get_default_crypto_metrics():
    """Return default crypto metrics when data is unavailable."""
    return {
        "24h_volume": 0.0,
        "24h_vwap": 0.0,
        "24h_open": 0.0,
        "24h_close": 0.0,
        "24h_high": 0.0,
        "24h_low": 0.0,
        "24h_transactions": 0,
        "return_on_equity": 0.0,
        "net_margin": 0.0,
        "operating_margin": 0.0,
        "revenue_growth": 0.0,
        "earnings_growth": 0.0,
        "book_value_growth": 0.0,
        "current_ratio": 0.0,
        "debt_to_equity": 0.0,
        "free_cash_flow_per_share": 0.0,
        "earnings_per_share": 0.0,
        "price_to_earnings_ratio": 0.0,
        "price_to_book_ratio": 0.0,
        "price_to_sales_ratio": 0.0
    }


def get_insider_trades(ticker: str, start_date: str, end_date: str, asset_type='stock'):
    """Fetch insider trades from Polygon.io.
    Note: This is only available for stocks, not crypto."""
    if asset_type.lower() == 'crypto':
        return []  # Crypto doesn't have insider trades
        
    headers = get_polygon_headers()
    url = "https://api.polygon.io/v2/reference/insider-transactions"
    params = {
        "ticker": ticker,
        "limit": 50
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error fetching insider trades: {response.status_code} - {response.text}")
        
    data = response.json()
    results = data.get("results", [])

    # Filter results by date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Transform Polygon data to match the expected format
    insider_trades = []
    for result in results:
        filing_date = datetime.strptime(result.get("filing_date", "1970-01-01"), '%Y-%m-%d')
        if start <= filing_date <= end:
            insider_trades.append({
                "filing_date": result.get("filing_date"),
                "transaction_date": result.get("transaction_date"),
                "insider_name": result.get("insider_name"),
                "insider_title": result.get("insider_title"),
                "transaction_type": result.get("transaction_type"),
                "shares": result.get("shares", 0),
                "share_price": result.get("share_price", 0),
                "total_value": result.get("shares", 0) * result.get("share_price", 0)
            })

    return insider_trades


def prices_to_df(prices):
    """Convert prices to a DataFrame."""
    if not prices:
        # Return empty DataFrame with correct columns if no prices
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.DataFrame(prices)
    df.set_index('date', inplace=True)
    return df


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


def get_price_data(ticker, start_date, end_date, asset_type='crypto'):
    """Get price data from Polygon API.
    
    Args:
        ticker (str): The ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        asset_type (str, optional): Type of asset ('crypto' or 'stock'). Defaults to 'crypto'.
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    return get_prices(ticker, start_date, end_date, asset_type=asset_type)


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
