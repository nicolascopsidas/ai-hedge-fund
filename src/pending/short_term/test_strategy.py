import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from src.short_term.models import TimeFrame, MarketSignal
from src.short_term.data_manager import MultiTimeframeData
from src.short_term.technicals import TechnicalAnalyzer
from src.short_term.signal_generator import SignalGenerator


class StrategyTester:
    def __init__(self, symbol: str = "BTC"):
        """Initialize the strategy tester.
        
        Args:
            symbol (str): The cryptocurrency symbol to test
        """
        self.symbol = symbol
        self.data_manager = MultiTimeframeData(symbol)
        self.technical_analyzer = TechnicalAnalyzer()
        self.signal_generator = SignalGenerator()
        
        # For storing test results
        self.signals: Dict[TimeFrame, List[MarketSignal]] = {
            tf: [] for tf in TimeFrame
        }
        
    def test_data_fetching(self, timeframe: TimeFrame) -> pd.DataFrame:
        """Test data fetching for a specific timeframe.
        
        Args:
            timeframe (TimeFrame): The timeframe to test
            
        Returns:
            pd.DataFrame: The fetched data
        """
        print(f"\nTesting data fetching for {timeframe.value}...")
        
        try:
            # Set date range for testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Get 30 days of data
            
            # Fetch data
            data = self.data_manager.get_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                print("❌ No data received")
                return pd.DataFrame()
                
            # Print data info
            print("✅ Data fetched successfully")
            print(f"Shape: {data.shape}")
            print("\nColumns:", data.columns.tolist())
            print("\nSample data:")
            print(data.tail().to_string())
            
            # Check for missing values
            missing = data.isnull().sum()
            if missing.any():
                print("\nMissing values:")
                print(missing[missing > 0])
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def test_technical_analysis(self, timeframe: TimeFrame) -> None:
        """Test technical analysis for a specific timeframe.
        
        Args:
            timeframe (TimeFrame): The timeframe to test
        """
        print(f"\nTesting technical analysis for {timeframe.value}...")
        
        try:
            # Get data with date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = self.data_manager.get_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                print("❌ No data available for analysis")
                return
            
            # Run analysis
            signal = self.technical_analyzer.analyze_timeframe(data, timeframe)
            
            # Store signal
            self.signals[timeframe].append(signal)
            
            # Print results
            print("\nAnalysis Results:")
            print(f"Direction: {signal.direction}")
            print(f"Strength: {signal.strength:.2f}")
            print("\nIndicators:")
            for name, value in signal.indicators.items():
                print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
            
        except Exception as e:
            print(f"❌ Error in technical analysis: {str(e)}")
    
    def test_signal_generation(self) -> None:
        """Test signal generation across all timeframes."""
        print("\nTesting signal generation...")
        
        try:
            # Set date range for testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Get 30 days of data
            
            for timeframe in TimeFrame:
                # Get data for timeframe
                data = self.data_manager.get_data(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data.empty:
                    print(f"❌ No data available for {timeframe.value}")
                    continue
                
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    data=data,
                    timeframe=timeframe
                )
                
                # Store signal
                self.signals[timeframe].append(signal)
                
                # Print results
                print(f"\n{timeframe.value} Signal:")
                print(f"Direction: {signal.direction}")
                print(f"Strength: {signal.strength:.2f}")
                print("\nIndicators:")
                for name, value in signal.indicators.items():
                    print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
                
        except Exception as e:
            print(f"❌ Error in signal generation: {str(e)}")
    
    def plot_signals(self, timeframe: TimeFrame) -> None:
        """Plot price data and signals for a specific timeframe.
        
        Args:
            timeframe (TimeFrame): The timeframe to plot
        """
        try:
            # Get data for plotting
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Plot last 30 days
            
            data = self.data_manager.get_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                print("❌ Error in plotting: No data available")
                return
                
            # Check if we have enough data for plotting
            required_columns = ['close', 'sma_values', 'ema_values', 'rsi_values']
            if not all(col in data.columns for col in required_columns):
                print("❌ Error in plotting: Missing required indicators")
                return
            
            # Remove NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
            
            # Plot price and indicators
            ax1.plot(data.index, data['close'], label='Price', color='blue')
            ax1.plot(data.index, data['sma_values'], label='SMA', color='orange')
            ax1.plot(data.index, data['ema_values'], label='EMA', color='green')
            
            # Plot RSI
            ax2.plot(data.index, data['rsi_values'], label='RSI', color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            
            # Add labels and title
            ax1.set_title(f'{self.symbol} - {timeframe.value} Timeframe')
            ax1.set_ylabel('Price')
            ax2.set_ylabel('RSI')
            
            # Add legends
            ax1.legend()
            ax2.legend()
            
            # Adjust layout and display
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error in plotting: {str(e)}")
    
    def plot_analysis(self, timeframe: TimeFrame, periods: int = 100) -> None:
        """Plot price data with indicators for visual verification.
        
        Args:
            timeframe (TimeFrame): The timeframe to plot
            periods (int): Number of periods to plot
        """
        try:
            data = self.data_manager.get_data(timeframe, periods)
            if data.empty:
                print("❌ No data available for plotting")
                return
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), height_ratios=[3, 1, 1])
            fig.suptitle(f'{self.symbol} Analysis - {timeframe.value}')
            
            # Plot price and indicators
            ax1.plot(data.index, data['close'], label='Price')
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                ax1.plot(data.index, data['bb_upper'], 'r--', label='BB Upper')
                ax1.plot(data.index, data['bb_lower'], 'r--', label='BB Lower')
            ax1.legend()
            ax1.set_title('Price and Bollinger Bands')
            
            # Plot RSI
            if 'rsi' in data.columns:
                ax2.plot(data.index, data['rsi'], label='RSI')
                ax2.axhline(y=70, color='r', linestyle='--')
                ax2.axhline(y=30, color='r', linestyle='--')
                ax2.legend()
                ax2.set_title('RSI')
            
            # Plot MACD
            if all(x in data.columns for x in ['macd', 'macd_signal']):
                ax3.plot(data.index, data['macd'], label='MACD')
                ax3.plot(data.index, data['macd_signal'], label='Signal')
                ax3.legend()
                ax3.set_title('MACD')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error in plotting: {str(e)}")
    
    def run_full_test(self) -> None:
        """Run a complete test of all components."""
        print("\n=== Starting Full Strategy Test ===")
        
        # Test data fetching for each timeframe
        for timeframe in TimeFrame:
            data = self.test_data_fetching(timeframe)
            if not data.empty:
                self.test_technical_analysis(timeframe)
                self.plot_analysis(timeframe)
        
        # Test signal generation
        self.test_signal_generation()
        
        print("\n=== Strategy Test Complete ===")


def main():
    """Run the strategy test."""
    print("\n=== Starting Full Strategy Test ===\n")
    
    tester = StrategyTester()
    
    # Test each timeframe
    for timeframe in TimeFrame:
        # Test data fetching
        data = tester.test_data_fetching(timeframe)
        
        # Test technical analysis if we have data
        if not data.empty:
            tester.test_technical_analysis(timeframe)
    
    # Test signal generation
    tester.test_signal_generation()
    
    print("\n=== Strategy Test Complete ===")


if __name__ == "__main__":
    main()
