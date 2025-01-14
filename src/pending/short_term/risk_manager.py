from datetime import timedelta
from typing import Dict, Optional, Tuple
import uuid

from src.short_term.models import TradeSetup, Position


class RiskManager:
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.1):
        """Initialize the risk manager.
        
        Args:
            max_risk_per_trade (float): Maximum risk per trade as decimal (0.02 = 2%)
            max_portfolio_risk (float): Maximum total portfolio risk (0.1 = 10%)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
        # Maximum leverage based on trade duration
        self.max_leverage = {
            timedelta(minutes=30): 5.0,  # Higher leverage for shorter trades
            timedelta(hours=4): 3.0,     # Medium leverage for medium duration
            timedelta(days=3): 2.0       # Lower leverage for longer trades
        }
        
        # Active positions
        self.positions: Dict[str, Position] = {}
    
    def _calculate_position_risk(self, 
                               entry_price: float, 
                               stop_loss: float) -> float:
        """Calculate the risk per unit of position."""
        return abs(entry_price - stop_loss) / entry_price
    
    def _get_max_leverage(self, duration: timedelta) -> float:
        """Get maximum allowed leverage based on trade duration."""
        # Find the closest duration in our leverage map
        closest_duration = min(
            self.max_leverage.keys(),
            key=lambda k: abs(k - duration)
        )
        return self.max_leverage[closest_duration]
    
    def _adjust_leverage_for_volatility(self, 
                                      base_leverage: float, 
                                      signal_strength: float) -> float:
        """Adjust leverage based on signal strength."""
        # Reduce leverage if signal strength is not optimal
        return base_leverage * max(0.5, signal_strength)
    
    def size_position(self, 
                     setup: TradeSetup, 
                     account_balance: float) -> Tuple[float, float]:
        """Calculate position size and leverage.
        
        Args:
            setup (TradeSetup): The trade setup
            account_balance (float): Current account balance
            
        Returns:
            Tuple[float, float]: (position_size, leverage)
        """
        # Calculate base position size without leverage
        risk_amount = account_balance * self.max_risk_per_trade
        position_risk = self._calculate_position_risk(
            setup.entry_price, setup.stop_loss
        )
        
        base_size = risk_amount / position_risk if position_risk > 0 else 0
        
        # Calculate leverage
        max_leverage = self._get_max_leverage(setup.expected_duration)
        adjusted_leverage = self._adjust_leverage_for_volatility(
            max_leverage, setup.signal_strength
        )
        
        # Ensure we don't exceed portfolio risk limits
        total_exposure = sum(pos.size * pos.leverage 
                           for pos in self.positions.values())
        max_new_exposure = account_balance * self.max_portfolio_risk - total_exposure
        
        # Adjust size if necessary
        leveraged_size = base_size * adjusted_leverage
        if leveraged_size * setup.entry_price > max_new_exposure:
            leveraged_size = max_new_exposure / setup.entry_price
            base_size = leveraged_size / adjusted_leverage
        
        return base_size, adjusted_leverage
    
    def create_position(self, 
                       setup: TradeSetup, 
                       account_balance: float) -> Optional[Position]:
        """Create a new position from a trade setup.
        
        Args:
            setup (TradeSetup): The trade setup
            account_balance (float): Current account balance
            
        Returns:
            Optional[Position]: The created position or None if invalid
        """
        # Check if we can take on more risk
        total_risk = sum(
            self._calculate_position_risk(pos.entry_price, pos.stop_loss) * 
            pos.size * pos.leverage / account_balance
            for pos in self.positions.values()
        )
        
        if total_risk >= self.max_portfolio_risk:
            return None
            
        # Calculate position size and leverage
        size, leverage = self.size_position(setup, account_balance)
        
        if size <= 0 or leverage <= 0:
            return None
            
        # Create position
        position = Position(
            trade_id=str(uuid.uuid4()),
            direction=setup.direction,
            entry_price=setup.entry_price,
            current_price=setup.entry_price,
            size=size,
            leverage=leverage,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            entry_time=setup.timestamp,
            expected_duration=setup.expected_duration,
            pnl=0.0,
            status="open"
        )
        
        # Store position
        self.positions[position.trade_id] = position
        
        return position
    
    def update_position(self, 
                       trade_id: str, 
                       current_price: float) -> Optional[str]:
        """Update position status and check for exit conditions.
        
        Args:
            trade_id (str): The position's trade ID
            current_price (float): Current market price
            
        Returns:
            Optional[str]: Exit reason if position should be closed, None otherwise
        """
        if trade_id not in self.positions:
            return None
            
        position = self.positions[trade_id]
        
        # Update current price and P&L
        position.current_price = current_price
        price_change = (current_price - position.entry_price) / position.entry_price
        position.pnl = price_change * position.size * position.leverage
        
        # Check stop loss
        if position.direction == "long" and current_price <= position.stop_loss:
            return "stop_loss"
        elif position.direction == "short" and current_price >= position.stop_loss:
            return "stop_loss"
            
        # Check take profit
        if position.direction == "long" and current_price >= position.take_profit:
            return "take_profit"
        elif position.direction == "short" and current_price <= position.take_profit:
            return "take_profit"
            
        # Check time-based exit
        if position.is_expired:
            return "time_exit"
            
        return None
    
    def close_position(self, trade_id: str, reason: str) -> None:
        """Close a position and record the reason.
        
        Args:
            trade_id (str): The position's trade ID
            reason (str): Reason for closing the position
        """
        if trade_id in self.positions:
            position = self.positions[trade_id]
            position.status = f"closed_{reason}"
            # In a real implementation, we would also execute the close order here
