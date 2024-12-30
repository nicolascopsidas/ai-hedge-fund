"""Utility functions for formatting data"""

from datetime import datetime
from typing import Dict, List, Union
import pandas as pd

def format_timestamp(timestamp: Union[datetime, int, float, str]) -> str:
    """Format timestamp to ISO 8601 format without milliseconds"""
    if isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp / 1000.0)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")

def format_price_data(price_data: pd.DataFrame) -> Dict[str, any]:
    """Format price data for output"""
    if price_data.empty:
        return {
            "current": {
                "timestamp": None,
                "value": None,
                "description": "Prix actuel de l'actif"
            },
            "historical": {
                "data": [],
                "description": "Prix historique sur les dernieres periodes (du plus ancien au plus recent)"
            }
        }
        
    latest_price = float(price_data["close"].iloc[-1])
    historical_data = price_data["close"].iloc[-5:]
    historical_prices = [
        {
            "timestamp": index.strftime("%Y-%m-%dT%H:%M:%S"),
            "value": round(float(value), 6)
        } for index, value in historical_data.items()
    ]
    
    # Make sure latest_price is the last price
    if historical_prices:
        historical_prices[-1]["value"] = round(float(latest_price), 6)
    
    return {
        "current": {
            "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
            "value": round(float(latest_price), 6),
            "description": "Prix actuel de l'actif"
        },
        "historical": {
            "data": historical_prices,
            "description": "Prix historique sur les dernieres periodes (du plus ancien au plus recent)"
        }
    }

def format_number(value: float, decimals: int = 2) -> float:
    """Format a number with specified decimal places"""
    if value is None:
        return None
    return round(float(value), decimals)
