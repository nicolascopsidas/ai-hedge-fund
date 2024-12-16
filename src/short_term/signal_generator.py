from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.short_term.models import TimeFrame, MarketSignal, TradeSetup
from src.short_term.data_manager import MultiTimeframeData
from src.short_term.technicals import TechnicalAnalyzer


class SignalGenerator:
    def __init__(self):
        """Initialize the signal generator."""
        self.technical_analyzer = TechnicalAnalyzer()
        self.required_confluence = 3  # Minimum number of confirming signals

        # Weight for each timeframe's contribution
        self.timeframe_weights = {
            TimeFrame.M5: 0.1,    # Quick signals but noisy
            TimeFrame.M15: 0.15,  # Entry/exit confirmation
            TimeFrame.H1: 0.3,    # Primary trend direction
            TimeFrame.H4: 0.25,   # Major support/resistance
            TimeFrame.D1: 0.2     # Overall market context
        }

        # Signal thresholds
        self.min_signal_strength = 0.6
        self.min_rr_ratio = 1.5
        self.max_risk_percent = 0.02  # 2% max risk per trade

        # Market condition filters
        self.volatility_threshold = 0.02  # 2% price movement
        self.volume_threshold = 1.5  # 50% above average

    def analyze_market(self,
                       mtf_data: MultiTimeframeData) -> Dict[TimeFrame, MarketSignal]:
        """Analyze all timeframes and generate signals."""
        try:
            signals = {}
            for timeframe in TimeFrame:
                data = mtf_data.get_data(
                    mtf_data.ticker,
                    timeframe,
                    datetime.now() - TimeFrame.get_lookback(timeframe),
                    datetime.now()
                )
                if not data.empty:
                    signals[timeframe] = self.technical_analyzer.analyze_timeframe(
                        data, timeframe
                    )
            return signals
        except Exception as e:
            print(f"Error analyzing market: {str(e)}")
            return {}

    def generate_signal(self, data, timeframe: TimeFrame) -> MarketSignal:
        """Generate a signal for a specific timeframe.

        Args:
            data (pd.DataFrame): The price data for analysis
            timeframe (TimeFrame): The timeframe being analyzed

        Returns:
            MarketSignal: The generated signal
        """
        try:
            return self.technical_analyzer.analyze_timeframe(data, timeframe)
        except Exception as e:
            print(f"Error generating signal for {timeframe}: {str(e)}")
            return MarketSignal(
                timeframe=timeframe,
                direction="neutral",
                strength=0.0,
                indicators={},
                timestamp=datetime.now()
            )

    def _calculate_signal_strength(self,
                                   signals: Dict[TimeFrame, MarketSignal]) -> float:
        """Calculate overall signal strength using weighted average."""
        try:
            total_weight = 0
            weighted_strength = 0

            for timeframe, signal in signals.items():
                weight = self.timeframe_weights[timeframe]
                # Adjust weight based on signal quality
                if signal.direction == "neutral":
                    weight *= 0.5
                weighted_strength += signal.strength * weight
                total_weight += weight

            return weighted_strength / total_weight if total_weight > 0 else 0
        except Exception as e:
            print(f"Error calculating signal strength: {str(e)}")
            return 0.0

    def _count_confluence(self,
                          signals: Dict[TimeFrame, MarketSignal],
                          direction: str) -> int:
        """Count how many timeframes confirm the signal direction."""
        try:
            # Weight confluence by timeframe importance
            confluence = 0
            for timeframe, signal in signals.items():
                if signal.direction == direction:
                    weight = self.timeframe_weights[timeframe]
                    confluence += weight
            return round(confluence * 5)  # Scale to roughly 0-5 range
        except Exception as e:
            print(f"Error counting confluence: {str(e)}")
            return 0

    def _estimate_trade_duration(self,
                                 signals: Dict[TimeFrame, MarketSignal]) -> timedelta:
        """Estimate optimal trade duration based on signals."""
        try:
            # Start with base duration
            base_duration = timedelta(hours=4)

            # Get signal strengths for different timeframes
            short_term_strength = max(
                signals[TimeFrame.M5].strength if TimeFrame.M5 in signals else 0,
                signals[TimeFrame.M15].strength if TimeFrame.M15 in signals else 0
            )
            medium_term_strength = signals[TimeFrame.H1].strength if TimeFrame.H1 in signals else 0
            long_term_strength = max(
                signals[TimeFrame.H4].strength if TimeFrame.H4 in signals else 0,
                signals[TimeFrame.D1].strength if TimeFrame.D1 in signals else 0
            )

            # Adjust duration based on signal strengths
            if short_term_strength > 0.8 and long_term_strength < 0.3:
                return timedelta(minutes=30)  # Very short trade
            elif long_term_strength > 0.8 and short_term_strength > 0.6:
                return timedelta(days=3)  # Longer trade
            elif medium_term_strength > 0.7:
                return timedelta(hours=8)  # Medium duration

            return base_duration

        except Exception as e:
            print(f"Error estimating trade duration: {str(e)}")
            return timedelta(hours=4)

    def _check_market_conditions(self,
                                 data: pd.DataFrame,
                                 signals: Dict[TimeFrame, MarketSignal]) -> bool:
        """Check if market conditions are suitable for trading."""
        try:
            # Check volatility
            recent_prices = data['close'].tail(20)
            volatility = (recent_prices.std() / recent_prices.mean())
            if volatility > self.volatility_threshold:
                print(f"Volatility too high: {volatility:.2%}")
                return False

            # Check volume
            recent_volume = data['volume'].tail(20).mean()
            avg_volume = data['volume'].mean()
            if recent_volume < avg_volume * self.volume_threshold:
                print(f"Volume too low: {
                      recent_volume / avg_volume:.2f}x average")
                return False

            # Check trend alignment
            h1_signal = signals.get(TimeFrame.H1)
            h4_signal = signals.get(TimeFrame.H4)
            if h1_signal and h4_signal:
                if h1_signal.direction != h4_signal.direction:
                    print("H1 and H4 trends not aligned")
                    return False

            return True

        except Exception as e:
            print(f"Error checking market conditions: {str(e)}")
            return False

    def _calculate_position_size(self,
                                 current_price: float,
                                 stop_loss: float,
                                 account_size: float = 10000.0) -> float:
        """Calculate position size based on risk parameters."""
        try:
            risk_amount = account_size * self.max_risk_percent
            price_risk = abs(current_price - stop_loss)
            position_size = risk_amount / price_risk
            return position_size
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0.0

    def generate_trade_setup(self,
                             mtf_data: MultiTimeframeData) -> Optional[TradeSetup]:
        """Generate a trade setup if conditions are favorable."""
        try:
            # Get signals for all timeframes
            signals = self.analyze_market(mtf_data)
            if not signals:
                return None

            # Get H1 data for market condition checks
            h1_data = mtf_data.get_data(
                mtf_data.ticker,
                TimeFrame.H1,
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            if h1_data.empty:
                return None

            # Check market conditions first
            if not self._check_market_conditions(h1_data, signals):
                return None

            # Determine primary direction from H1 timeframe
            primary_signal = signals.get(TimeFrame.H1)
            if not primary_signal or primary_signal.direction == "neutral":
                return None

            # Check confluence
            confluence_count = self._count_confluence(
                signals, primary_signal.direction)
            if confluence_count < self.required_confluence:
                return None

            # Calculate overall signal strength
            signal_strength = self._calculate_signal_strength(signals)
            if signal_strength < self.min_signal_strength:
                return None

            # Get current price and ATR for stop loss calculation
            current_price = mtf_data.get_latest_price(TimeFrame.M5)
            atr = self.technical_analyzer._calculate_atr(h1_data, 14).iloc[-1]

            # Calculate stop loss and take profit
            if primary_signal.direction == "long":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:  # short
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)

            # Check risk/reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if reward / risk < self.min_rr_ratio:
                return None

            # Calculate position size
            size = self._calculate_position_size(current_price, stop_loss)
            if size <= 0:
                return None

            # Estimate trade duration
            expected_duration = self._estimate_trade_duration(signals)

            return TradeSetup(
                direction=primary_signal.direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=size,
                leverage=1.0,  # Base leverage, to be adjusted by risk manager
                expected_duration=expected_duration,
                timeframe_confluence=confluence_count,
                signal_strength=signal_strength,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"Error generating trade setup: {str(e)}")
            return None
