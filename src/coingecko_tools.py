import os
from datetime import datetime
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class CoinGeckoAPI:
    """CoinGecko API wrapper for cryptocurrency data retrieval."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    TICKER_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "MATIC": "polygon",
    }

    def __init__(self):
        """Initialize the CoinGecko API client."""
        self.session = requests.Session()
        self.api_key = os.getenv("COINGECKO_API_KEY")

    def _get_headers(self) -> Dict:
        """Get headers for API requests."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "AI Hedge Fund/1.0"
        }
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key
        return headers

    def _get_coin_id(self, ticker: str) -> str:
        """Convert ticker to CoinGecko coin ID."""
        return self.TICKER_MAP.get(ticker.upper(), ticker.lower())

    def get_key_metrics(self, ticker: str) -> Optional[Dict]:
        """Get essential metrics for trading decisions.
        
        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC', 'ETH')

        Returns:
            Dict containing key trading metrics:
            - Price metrics (current, 24h/7d changes, volatility)
            - Market quality (market cap, volume, rank)
            - Risk metrics (supply ratio, developer activity)
            - Sentiment indicators (social metrics, community sentiment)
            
            Returns None if data retrieval fails.
        """
        coin_id = self._get_coin_id(ticker)

        try:
            response = self.session.get(
                f"{self.BASE_URL}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "true",
                    "market_data": "true",
                    "community_data": "true",
                    "developer_data": "true",
                    "sparkline": "false"
                },
                headers=self._get_headers()
            )
            
            if response.status_code != 200:
                print(f"Error fetching data for {ticker}: {response.status_code}")
                return None

            data = response.json()
            market_data = data.get('market_data', {})
            
            # Calculate volatility
            high_24h = market_data.get('high_24h', {}).get('usd', 0)
            low_24h = market_data.get('low_24h', {}).get('usd', 0)
            volatility_24h = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0

            # Calculate supply ratio
            max_supply = market_data.get('max_supply')
            circulating_supply = market_data.get('circulating_supply')
            supply_ratio = (circulating_supply / max_supply * 100) if max_supply and circulating_supply else None

            return {
                # Price Action
                "price_usd": market_data.get('current_price', {}).get('usd', 0),
                "change_24h": market_data.get('price_change_percentage_24h', 0),
                "change_7d": market_data.get('price_change_percentage_7d', 0),
                "volatility_24h": volatility_24h,
                
                # Market Quality
                "market_cap_usd": market_data.get('market_cap', {}).get('usd', 0),
                "volume_24h_usd": market_data.get('total_volume', {}).get('usd', 0),
                "market_cap_rank": data.get('market_cap_rank'),
                
                # Risk Metrics
                "supply_ratio": supply_ratio,
                "dev_commits_4w": data.get('developer_data', {}).get('commit_count_4_weeks', 0),
                "dev_stars": data.get('developer_data', {}).get('stars', 0),
                
                # Sentiment Indicators
                "twitter_followers": data.get('community_data', {}).get('twitter_followers', 0),
                "sentiment_votes_up_percentage": data.get('sentiment_votes_up_percentage', 0),
                
                # Timestamp
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            return None

    def get_historical_data(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """Get historical market data for a cryptocurrency.

        Args:
            ticker (str): Cryptocurrency ticker (e.g., 'BTC', 'ETH')
            days (int): Number of days of historical data (max 365)

        Returns:
            Dict containing historical prices, market caps, and volumes.
            Returns None if data retrieval fails.
        """
        coin_id = self._get_coin_id(ticker)

        try:
            response = self.session.get(
                f"{self.BASE_URL}/coins/{coin_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": min(days, 365),
                    "interval": "daily"
                },
                headers=self._get_headers()
            )

            if response.status_code != 200:
                print(f"Error fetching historical data for {ticker}: {response.status_code}")
                return None

            return response.json()

        except Exception as e:
            print(f"Error processing historical data for {ticker}: {str(e)}")
            return None
