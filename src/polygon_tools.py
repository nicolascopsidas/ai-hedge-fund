import os
from datetime import datetime
import requests
import pandas as pd


def get_polygon_headers():
    """Get headers for Polygon API requests."""
    return {
        "Authorization": f"Bearer {os.environ.get('POLYGON_API_KEY')}"
    }


def format_crypto_ticker(ticker: str) -> str:
    """Format crypto ticker for Polygon API (e.g., 'BTC' -> 'X:BTCUSD')"""
    if ticker.startswith('X:') or ticker.endswith('USD'):
        return ticker
    return f"X:{ticker}USD"


def get_prices(ticker: str, start_date: str, end_date: str, asset_type: str = 'stock'):
    """Fetch price data from Polygon.io.
    
    Args:
        ticker: The ticker symbol (e.g., 'AAPL' for stocks, 'BTC' for crypto)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        asset_type: Either 'stock' or 'crypto'
    """
    headers = get_polygon_headers()
    
    # Format ticker for crypto if needed
    if asset_type.lower() == 'crypto':
        ticker = format_crypto_ticker(ticker)
    
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/"
        f"range/1/day/{start_date}/{end_date}"
    )
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching price data: {response.status_code} - {response.text}")
        return []

    data = response.json()
    results = data.get("results", [])

    # Transform Polygon data to match the expected format
    prices = []
    for result in results:
        try:
            prices.append({
                "date": datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'),
                "open": float(result['o']),
                "high": float(result['h']),
                "low": float(result['l']),
                "close": float(result['c']),
                "volume": float(result['v'])
            })
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error processing price data: {e}")
            continue

    return sorted(prices, key=lambda x: x['date'])


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
        return [get_default_metrics()]

    data = response.json()
    results = data.get("results", [])

    if not results:
        return [get_default_metrics()]

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
        return [get_default_crypto_metrics()]
        
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
        return []

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


def calculate_macd(prices_df):
    """Calculate MACD."""
    exp1 = prices_df['close'].ewm(span=12, adjust=False).mean()
    exp2 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def calculate_rsi(prices_df, period=14):
    """Calculate RSI."""
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices_df, window=20):
    """Calculate Bollinger Bands."""
    sma = prices_df['close'].rolling(window=window).mean()
    std = prices_df['close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band


def calculate_obv(prices_df):
    """Calculate On-Balance Volume."""
    obv = (prices_df['volume'] *
           (~prices_df['close'].diff().le(0) * 2 - 1)).cumsum()
    return obv


def get_price_data(ticker, start_date, end_date, asset_type='crypto'):
    """Get price data and convert to DataFrame."""
    prices = get_prices(ticker, start_date, end_date, asset_type=asset_type)
    return prices_to_df(prices)


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
