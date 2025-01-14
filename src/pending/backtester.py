from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from src.polygon_tools import get_price_data
from src.agents import run_hedge_fund


class Backtester:
    """
    Backtester for cryptocurrency trading strategies.
    
    Args:
        agent: Trading agent that implements the strategy
        ticker (str): Cryptocurrency ticker (e.g., 'BTC')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_capital (float): Initial capital in quote currency
        lookback_days (int, optional): Number of days to look back for analysis. Defaults to 30
    
    Raises:
        ValueError: If start_date is after end_date or dates are invalid
    """
    
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, lookback_days=30):
        # Validate dates
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start >= end:
                raise ValueError("start_date must be before end_date")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")
            
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.lookback_days = lookback_days
        
        # Initialize portfolio with cash and crypto assets
        self.portfolio = {
            "cash": initial_capital,
            "assets": 0,
            "portfolio_value": initial_capital
        }
        self.portfolio_values = []

    def parse_action(self, agent_output):
        """Parse the trading action from agent output."""
        try:
            action = agent_output.get('action', 'hold').lower()
            quantity = float(agent_output.get('quantity', 0))
            
            if action not in ['buy', 'sell', 'hold']:
                print(f"Invalid action '{action}', defaulting to 'hold'")
                return 'hold', 0
                
            return action, quantity
        except (ValueError, AttributeError) as e:
            print(f"Error parsing agent output: {e}")
            return 'hold', 0

    def execute_trade(self, action, quantity, current_price):
        """
        Execute a trade based on the action and validate against portfolio constraints.
        
        Returns:
            float: Actually executed quantity
        """
        if action == 'hold' or quantity <= 0:
            return 0

        if action == 'buy':
            max_possible = self.portfolio['cash'] / current_price
            executable_quantity = min(quantity, max_possible)
            
            if executable_quantity < quantity:
                print(f"Insufficient funds. Reduced buy order from {quantity} to {executable_quantity}")
                
            cost = executable_quantity * current_price
            self.portfolio['cash'] -= cost
            self.portfolio['assets'] += executable_quantity
            return executable_quantity

        elif action == 'sell':
            executable_quantity = min(quantity, self.portfolio['assets'])
            
            if executable_quantity < quantity:
                print(f"Insufficient assets. Reduced sell order from {quantity} to {executable_quantity}")
                
            proceeds = executable_quantity * current_price
            self.portfolio['cash'] += proceeds
            self.portfolio['assets'] -= executable_quantity
            return -executable_quantity

        return 0

    def run_backtest(self):
        """Run the backtest simulation."""
        print("Starting backtest...")
        print(f"{'Date':11} {'Ticker':6} {'Action':6} {'Quantity':8} {'Price':11} {'Cash':8} {'Assets':6} {'Total Value'}")
        print("-" * 70)

        # Generate dates for the backtest period
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='D')

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            agent_output = self.agent(
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio
            )

            action, quantity = self.parse_action(agent_output)
            df = get_price_data(self.ticker, lookback_start, current_date_str)
            
            if df.empty:
                print(f"No price data available for {self.ticker} on {current_date_str}")
                continue
                
            current_price = df.iloc[-1]['close']

            # Execute the trade with validation
            executed_quantity = self.execute_trade(action, quantity, current_price)

            # Update total portfolio value
            self.portfolio['portfolio_value'] = self.portfolio['cash'] + self.portfolio['assets'] * current_price

            # Log the current state with executed quantity
            print(
                f"{current_date.strftime('%Y-%m-%d'):<11} {self.ticker:<6} {action:<6} {executed_quantity:>8} {current_price:>11.2f} "
                f"{self.portfolio['cash']:>8.2f} {self.portfolio['assets']:>6} {self.portfolio['portfolio_value']:>12.2f}"
            )

            # Record the portfolio value
            self.portfolio_values.append(
                {"Date": current_date, "Portfolio Value": self.portfolio['portfolio_value']}
            )

    def analyze_performance(self):
        # Convert portfolio values to DataFrame
        performance_df = pd.DataFrame(self.portfolio_values).set_index("Date")

        # Calculate total return
        total_return = (
                           self.portfolio["portfolio_value"] - self.initial_capital
                       ) / self.initial_capital
        print(f"Total Return: {total_return * 100:.2f}%")

        # Plot the portfolio value over time
        performance_df["Portfolio Value"].plot(
            title="Portfolio Value Over Time", figsize=(12, 6)
        )
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Date")
        plt.show()

        # Compute daily returns
        performance_df["Daily Return"] = performance_df["Portfolio Value"].pct_change()

        # Calculate Sharpe Ratio (assuming 252 trading days in a year)
        mean_daily_return = performance_df["Daily Return"].mean()
        std_daily_return = performance_df["Daily Return"].std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * (252 ** 0.5)
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Calculate Maximum Drawdown
        rolling_max = performance_df["Portfolio Value"].cummax()
        drawdown = performance_df["Portfolio Value"] / rolling_max - 1
        max_drawdown = drawdown.min()
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

        return performance_df
    
### 4. Run the Backtest #####
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cryptocurrency trading strategy backtest")
    parser.add_argument("--ticker", type=str, required=True, help="Cryptocurrency ticker (e.g., 'BTC')")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial_capital", type=float, default=10000, help="Initial capital in USD")
    parser.add_argument("--lookback_days", type=int, default=30, help="Number of days to look back for analysis")
    
    args = parser.parse_args()
    
    # Run the backtest with the specified parameters
    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        lookback_days=args.lookback_days
    )
    
    # Run the backtest
    backtester.run_backtest()
    
    # Analyze and display results
    performance_df = backtester.analyze_performance()
