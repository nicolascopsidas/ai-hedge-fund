import requests
from datetime import datetime
from typing import Dict, List, Optional


class FearGreedIndex:
    """Fear and Greed Index API wrapper for crypto market sentiment analysis."""
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        """Initialize the Fear and Greed Index client."""
        self.session = requests.Session()
        
    def get_current_index(self) -> Optional[Dict]:
        """Get the current Fear and Greed Index.
        
        Returns:
            Dict containing current index value and classification:
            - value: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
            - value_classification: Textual representation of the value
            - timestamp: ISO format timestamp
            Returns None if request fails.
        """
        try:
            response = self.session.get(
                self.BASE_URL,
                params={"limit": "1"}
            )
            
            if response.status_code != 200:
                print(f"Error fetching Fear & Greed Index: {response.status_code}")
                return None
                
            data = response.json()
            if not data.get("data"):
                return None
                
            current = data["data"][0]
            return {
                "value": int(current["value"]),
                "value_classification": current["value_classification"],
                "timestamp": datetime.fromtimestamp(int(current["timestamp"])).isoformat()
            }
            
        except Exception as e:
            print(f"Error processing Fear & Greed Index: {str(e)}")
            return None
            
    def get_historical_index(self, days: int = 30) -> Optional[List[Dict]]:
        """Get historical Fear and Greed Index data.
        
        Args:
            days (int): Number of days of historical data (default: 30)
            
        Returns:
            List of dicts containing historical index values:
            - value: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
            - value_classification: Textual representation of the value
            - timestamp: ISO format timestamp
            Returns None if request fails.
        """
        try:
            response = self.session.get(
                self.BASE_URL,
                params={"limit": str(days)}
            )
            
            if response.status_code != 200:
                print(f"Error fetching historical Fear & Greed Index: {response.status_code}")
                return None
                
            data = response.json()
            if not data.get("data"):
                return None
                
            return [{
                "value": int(entry["value"]),
                "value_classification": entry["value_classification"],
                "timestamp": datetime.fromtimestamp(int(entry["timestamp"])).isoformat()
            } for entry in data["data"]]
            
        except Exception as e:
            print(f"Error processing historical Fear & Greed Index: {str(e)}")
            return None
            
    def get_market_sentiment(self) -> str:
        """Get a simple market sentiment indicator based on the current index.
        
        Returns:
            String indicating market sentiment: 'bearish', 'neutral', or 'bullish'
            Returns 'unknown' if data cannot be retrieved.
        """
        current = self.get_current_index()
        if not current:
            return "unknown"
            
        value = current["value"]
        if value <= 25:  # Extreme Fear
            return "bearish"
        elif value <= 45:  # Fear
            return "bearish"
        elif value <= 55:  # Neutral
            return "neutral"
        elif value <= 75:  # Greed
            return "bullish"
        else:  # Extreme Greed
            return "bullish"
