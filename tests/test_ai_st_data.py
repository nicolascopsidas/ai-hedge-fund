import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from src.ai_st_bot.ai_st_data import AISTDataFetcher

class TestAISTDataFetcher(unittest.TestCase):
    def setUp(self):
        with patch.dict('os.environ', {'POLYGON_API_KEY': 'test_key'}):
            self.fetcher = AISTDataFetcher()
    
    def test_format_ticker(self):
        """Test cryptocurrency ticker formatting."""
        test_cases = [
            ('BTC', 'X:BTCUSD'),
            ('btc', 'X:BTCUSD'),
            ('ETH-USD', 'X:ETHUSD'),
            ('X:BTCUSD', 'X:BTCUSD'),
        ]
        for input_ticker, expected in test_cases:
            self.assertEqual(self.fetcher._format_ticker(input_ticker), expected)
    
    @patch('requests.get')
    def test_get_bars(self, mock_get):
        """Test fetching price bars."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {
                    'timestamp': 1609459200000,  # 2021-01-01
                    'open': 29000.0,
                    'high': 29500.0,
                    'low': 28500.0,
                    'close': 29200.0,
                    'volume': 1000.0,
                    'vwap': 29100.0,
                    'n': 100
                }
            ]
        }
        mock_get.return_value = mock_response
        
        df = self.fetcher.get_bars('BTC')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.index[0], pd.Timestamp('2021-01-01'))
    
    @patch('requests.get')
    def test_get_indicators(self, mock_get):
        """Test fetching technical indicators."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {
                    'timestamp': 1609459200000,
                    'value': 29100.0
                }
            ]
        }
        mock_get.return_value = mock_response
        
        indicators = self.fetcher.get_indicators('BTC')
        
        self.assertIsInstance(indicators, dict)
        self.assertTrue(all(indicator in indicators for indicator in ['sma', 'ema', 'rsi', 'macd']))
    
    @patch('requests.get')
    def test_get_market_data(self, mock_get):
        """Test fetching both price data and indicators."""
        # Mock successful API responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {
                    'timestamp': 1609459200000,
                    'open': 29000.0,
                    'high': 29500.0,
                    'low': 28500.0,
                    'close': 29200.0,
                    'volume': 1000.0,
                    'vwap': 29100.0,
                    'n': 100
                }
            ]
        }
        mock_get.return_value = mock_response
        
        bars, indicators = self.fetcher.get_market_data('BTC')
        
        self.assertIsInstance(bars, pd.DataFrame)
        self.assertIsInstance(indicators, dict)
        self.assertEqual(len(bars), 1)
        self.assertTrue(all(indicator in indicators for indicator in ['sma', 'ema', 'rsi', 'macd']))

if __name__ == '__main__':
    unittest.main()
