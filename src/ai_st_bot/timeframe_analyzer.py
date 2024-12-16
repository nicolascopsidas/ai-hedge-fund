from ai_st_data import AISTDataFetcher
import pandas as pd
from typing import Dict, List, Tuple
from enum import Enum

class Trend(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class TimeframeAnalyzer:
    def __init__(self):
        self.fetcher = AISTDataFetcher()
        
        # Define optimal timeframe configurations
        self.timeframe_configs = {
            "5m": {
                'timespan': "minute",
                'multiplier': 5,
                'rsi_period': 10,
                'ema_period': 20,
                'sma_period': 20,
                'macd_fast': 8,
                'macd_slow': 17,
                'macd_signal': 9
            },
            "15m": {
                'timespan': "minute",
                'multiplier': 15,
                'rsi_period': 14,
                'ema_period': 35,
                'sma_period': 35,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            "1h": {
                'timespan': "hour",
                'multiplier': 1,
                'rsi_period': 14,
                'ema_period': 50,
                'sma_period': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            "4h": {
                'timespan': "hour",
                'multiplier': 4,
                'rsi_period': 14,
                'ema_period': 50,
                'sma_period': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            "1d": {
                'timespan': "day",
                'multiplier': 1,
                'rsi_period': 21,
                'ema_period': 100,
                'sma_period': 100,
                'macd_fast': 19,
                'macd_slow': 39,
                'macd_signal': 9
            }
        }
    
    def calculate_confidence(self, signals: List[Tuple[str, float]], indicators: Dict, timeframe: str) -> float:
        """Calculate confidence score based on multiple factors.
        
        Args:
            signals: List of (indicator_name, signal_value) tuples
            indicators: Dictionary of indicator data
            timeframe: Current timeframe being analyzed
            
        Returns:
            Confidence score (0-100)
        """
        weights = {
            'signal_agreement': 0.4,
            'signal_strength': 0.3,
            'trend_consistency': 0.3
        }
        
        scores = {}
        
        # 1. Signal Agreement Score
        signal_values = [s[1] for s in signals]
        if signal_values:
            max_diff = max(signal_values) - min(signal_values)
            scores['signal_agreement'] = (1 - max_diff/2) * 100
        else:
            scores['signal_agreement'] = 0
            
        # 2. Signal Strength Score
        if signal_values:
            # Convert signal values to absolute strength
            abs_strengths = [abs(v) for v in signal_values]
            avg_strength = sum(abs_strengths) / len(abs_strengths)
            scores['signal_strength'] = avg_strength * 100
        else:
            scores['signal_strength'] = 0
            
        # 3. Trend Consistency Score
        trend_score = 0
        count = 0
        
        # Check EMA trend consistency
        if 'ema' in indicators and not indicators['ema'].empty:
            ema_values = indicators['ema']['value']
            if len(ema_values) >= 3:
                # Calculate consecutive moves in same direction
                moves = [1 if ema_values.iloc[i] > ema_values.iloc[i+1] else -1 
                        for i in range(len(ema_values)-1)]
                consistency = sum(1 for i in range(len(moves)-1) 
                                if moves[i] == moves[i+1]) / (len(moves)-1)
                trend_score += consistency * 100
                count += 1
        
        # Check RSI trend consistency
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi_values = indicators['rsi']['value']
            if len(rsi_values) >= 3:
                moves = [1 if rsi_values.iloc[i] > rsi_values.iloc[i+1] else -1 
                        for i in range(len(rsi_values)-1)]
                consistency = sum(1 for i in range(len(moves)-1) 
                                if moves[i] == moves[i+1]) / (len(moves)-1)
                trend_score += consistency * 100
                count += 1
        
        # Check SMA trend consistency
        if 'sma' in indicators and not indicators['sma'].empty:
            sma_values = indicators['sma']['value']
            if len(sma_values) >= 3:
                moves = [1 if sma_values.iloc[i] > sma_values.iloc[i+1] else -1 
                        for i in range(len(sma_values)-1)]
                consistency = sum(1 for i in range(len(moves)-1) 
                                if moves[i] == moves[i+1]) / (len(moves)-1)
                trend_score += consistency * 100
                count += 1
        
        scores['trend_consistency'] = trend_score / max(1, count)
        
        # Calculate weighted average
        final_confidence = sum(scores[k] * weights[k] for k in weights)
        
        return min(100, max(0, final_confidence))

    def analyze_trend_strength(self, indicators: Dict, timeframe: str) -> Dict:
        """Analyze trend strength using multiple indicators.
        
        Args:
            indicators: Dictionary containing indicator data
            timeframe: Timeframe being analyzed
            
        Returns:
            Dictionary containing trend analysis
        """
        strength = {
            'trend': Trend.NEUTRAL,
            'strength': 0,  # 0 to 100
            'confidence': 0  # 0 to 100
        }
        
        signals = []
        
        # Analyze RSI
        if 'rsi' in indicators and not indicators['rsi'].empty:
            rsi = indicators['rsi']['value'].iloc[0]
            if rsi > 70:
                signals.append(('rsi', 1.0))  # Strong bullish
            elif rsi > 60:
                signals.append(('rsi', 0.5))  # Moderate bullish
            elif rsi < 30:
                signals.append(('rsi', -1.0))  # Strong bearish
            elif rsi < 40:
                signals.append(('rsi', -0.5))  # Moderate bearish
            else:
                signals.append(('rsi', 0))  # Neutral
        
        # Analyze MACD
        if 'macd' in indicators and not indicators['macd'].empty:
            macd_data = indicators['macd'].iloc[0]
            histogram = macd_data['histogram']
            signal = macd_data['signal']
            value = macd_data['value']
            
            # MACD Line crossing
            if value > signal:
                signals.append(('macd_cross', 0.5))
            else:
                signals.append(('macd_cross', -0.5))
            
            # MACD Histogram
            if histogram > 0 and histogram > indicators['macd']['histogram'].iloc[1]:
                signals.append(('macd_hist', 1.0))  # Strong bullish
            elif histogram > 0:
                signals.append(('macd_hist', 0.5))  # Moderate bullish
            elif histogram < 0 and histogram < indicators['macd']['histogram'].iloc[1]:
                signals.append(('macd_hist', -1.0))  # Strong bearish
            elif histogram < 0:
                signals.append(('macd_hist', -0.5))  # Moderate bearish
        
        # Analyze Moving Averages (EMA and SMA)
        if 'ema' in indicators and 'sma' in indicators and not indicators['ema'].empty and not indicators['sma'].empty:
            ema_values = indicators['ema']['value']
            sma_values = indicators['sma']['value']
            
            if len(ema_values) >= 3 and len(sma_values) >= 3:
                # Calculate trends
                ema_trend = (ema_values.iloc[0] - ema_values.iloc[2]) / ema_values.iloc[2] * 100
                sma_trend = (sma_values.iloc[0] - sma_values.iloc[2]) / sma_values.iloc[2] * 100
                
                # EMA signals
                if ema_trend > 1:
                    signals.append(('ema', 1.0))  # Strong bullish
                elif ema_trend > 0.2:
                    signals.append(('ema', 0.5))  # Moderate bullish
                elif ema_trend < -1:
                    signals.append(('ema', -1.0))  # Strong bearish
                elif ema_trend < -0.2:
                    signals.append(('ema', -0.5))  # Moderate bearish
                else:
                    signals.append(('ema', 0))  # Neutral
                
                # SMA signals
                if sma_trend > 1:
                    signals.append(('sma', 1.0))  # Strong bullish
                elif sma_trend > 0.2:
                    signals.append(('sma', 0.5))  # Moderate bullish
                elif sma_trend < -1:
                    signals.append(('sma', -1.0))  # Strong bearish
                elif sma_trend < -0.2:
                    signals.append(('sma', -0.5))  # Moderate bearish
                else:
                    signals.append(('sma', 0))  # Neutral
                
                # MA Cross signals
                if ema_values.iloc[0] > sma_values.iloc[0] and ema_values.iloc[1] <= sma_values.iloc[1]:
                    signals.append(('ma_cross', 1.0))  # Golden cross
                elif ema_values.iloc[0] < sma_values.iloc[0] and ema_values.iloc[1] >= sma_values.iloc[1]:
                    signals.append(('ma_cross', -1.0))  # Death cross
                elif ema_values.iloc[0] > sma_values.iloc[0]:
                    signals.append(('ma_cross', 0.5))  # Bullish alignment
                else:
                    signals.append(('ma_cross', -0.5))  # Bearish alignment
        
        # Calculate overall strength
        if signals:
            total_strength = sum(signal[1] for signal in signals)
            avg_strength = total_strength / len(signals)
            
            # Convert to 0-100 scale
            strength['strength'] = (avg_strength + 1) * 50
            
            # Determine trend
            if avg_strength > 0.6:
                strength['trend'] = Trend.STRONG_BULLISH
            elif avg_strength > 0.2:
                strength['trend'] = Trend.BULLISH
            elif avg_strength < -0.6:
                strength['trend'] = Trend.STRONG_BEARISH
            elif avg_strength < -0.2:
                strength['trend'] = Trend.BEARISH
            else:
                strength['trend'] = Trend.NEUTRAL
            
            # Calculate confidence using new method
            strength['confidence'] = self.calculate_confidence(signals, indicators, timeframe)
        
        return strength
    
    def analyze_timeframes(self, ticker: str) -> Dict:
        """Analyze multiple timeframes and their relationships."""
        results = {}
        
        # Analyze each timeframe
        for tf_label, config in self.timeframe_configs.items():
            indicators = self.fetcher.get_indicators(
                ticker,
                timespan=config['timespan'],
                multiplier=config['multiplier'],
                limit=10,
                rsi_period=config['rsi_period'],
                ema_period=config['ema_period'],
                sma_period=config['sma_period'],
                macd_fast=config['macd_fast'],
                macd_slow=config['macd_slow'],
                macd_signal=config['macd_signal']
            )
            
            results[tf_label] = {
                'indicators': indicators,
                'analysis': self.analyze_trend_strength(indicators, tf_label)
            }
        
        return results
    
    def get_market_structure(self, results: Dict) -> Dict:
        """Analyze market structure using multiple timeframe analysis."""
        structure = {
            'primary_trend': None,
            'secondary_trend': None,
            'entry_trend': None,
            'confluence_level': 0,
            'trading_bias': Trend.NEUTRAL,
            'confidence': 0
        }
        
        # Primary trend (4h)
        if '4h' in results:
            structure['primary_trend'] = results['4h']['analysis']['trend']
        
        # Secondary trend (1h)
        if '1h' in results:
            structure['secondary_trend'] = results['1h']['analysis']['trend']
        
        # Entry trend (15m)
        if '15m' in results:
            structure['entry_trend'] = results['15m']['analysis']['trend']
        
        # Calculate confluence
        trends = [
            results[tf]['analysis']['trend'] 
            for tf in ['4h', '1h', '15m'] 
            if tf in results
        ]
        
        bullish_count = sum(1 for t in trends if t in [Trend.BULLISH, Trend.STRONG_BULLISH])
        bearish_count = sum(1 for t in trends if t in [Trend.BEARISH, Trend.STRONG_BEARISH])
        
        # Calculate confluence level (0-100)
        max_count = max(bullish_count, bearish_count)
        structure['confluence_level'] = (max_count / len(trends)) * 100
        
        # Determine overall bias
        if bullish_count > bearish_count:
            structure['trading_bias'] = Trend.BULLISH
        elif bearish_count > bullish_count:
            structure['trading_bias'] = Trend.BEARISH
        else:
            structure['trading_bias'] = Trend.NEUTRAL
        
        # Calculate overall confidence
        confidences = [
            results[tf]['analysis']['confidence']
            for tf in ['4h', '1h', '15m']
            if tf in results
        ]
        structure['confidence'] = sum(confidences) / len(confidences)
        
        return structure

def main():
    analyzer = TimeframeAnalyzer()
    
    print("Analyzing BTC market structure...")
    results = analyzer.analyze_timeframes("BTC")
    
    # Print detailed analysis for each timeframe
    for timeframe, data in results.items():
        print(f"\n{timeframe} Analysis:")
        print("=" * 50)
        analysis = data['analysis']
        
        # Print trend and overall metrics
        print(f"Overall Trend: {analysis['trend'].value}")
        print(f"Trend Strength: {analysis['strength']:.1f}")
        print(f"Final Confidence: {analysis['confidence']:.1f}%")
        
        # Print individual indicators
        print("\nIndicator Signals:")
        print("-" * 30)
        
        # RSI Analysis
        if 'rsi' in data['indicators']:
            rsi_value = data['indicators']['rsi']['value'].iloc[0]
            print(f"RSI:")
            print(f"  Current Value: {rsi_value:.1f}")
            print(f"  Interpretation: ", end="")
            if rsi_value > 70: print("Strongly Overbought")
            elif rsi_value > 60: print("Overbought")
            elif rsi_value < 30: print("Strongly Oversold")
            elif rsi_value < 40: print("Oversold")
            else: print("Neutral")
        
        # MACD Analysis
        if 'macd' in data['indicators']:
            macd_data = data['indicators']['macd'].iloc[0]
            print(f"\nMACD:")
            print(f"  MACD Line: {macd_data['value']:.4f}")
            print(f"  Signal Line: {macd_data['signal']:.4f}")
            print(f"  Histogram: {macd_data['histogram']:.4f}")
            print(f"  Signal: {'Bullish' if macd_data['value'] > macd_data['signal'] else 'Bearish'}")
        
        # Moving Averages Analysis
        if 'ema' in data['indicators'] and 'sma' in data['indicators']:
            ema_values = data['indicators']['ema']['value']
            sma_values = data['indicators']['sma']['value']
            if len(ema_values) >= 3 and len(sma_values) >= 3:
                ema_trend = (ema_values.iloc[0] - ema_values.iloc[2]) / ema_values.iloc[2] * 100
                sma_trend = (sma_values.iloc[0] - sma_values.iloc[2]) / sma_values.iloc[2] * 100
                
                print(f"\nMoving Averages:")
                print(f"  EMA Current: {ema_values.iloc[0]:.2f}")
                print(f"  EMA Previous: {ema_values.iloc[1]:.2f}")
                print(f"  EMA Trend: {ema_trend:.2f}%")
                print(f"  SMA Current: {sma_values.iloc[0]:.2f}")
                print(f"  SMA Previous: {sma_values.iloc[1]:.2f}")
                print(f"  SMA Trend: {sma_trend:.2f}%")
                
                # MA Cross Analysis
                print("  MA Cross Status: ", end="")
                if ema_values.iloc[0] > sma_values.iloc[0] and ema_values.iloc[1] <= sma_values.iloc[1]:
                    print("Golden Cross (Bullish)")
                elif ema_values.iloc[0] < sma_values.iloc[0] and ema_values.iloc[1] >= sma_values.iloc[1]:
                    print("Death Cross (Bearish)")
                elif ema_values.iloc[0] > sma_values.iloc[0]:
                    print("Bullish Alignment")
                else:
                    print("Bearish Alignment")
        
        # Print confidence breakdown
        print("\nConfidence Breakdown:")
        print("-" * 30)
        signals = []
        if 'rsi' in data['indicators']:
            rsi = data['indicators']['rsi']['value'].iloc[0]
            if rsi > 70: signals.append(('rsi', 1.0))
            elif rsi > 60: signals.append(('rsi', 0.5))
            elif rsi < 30: signals.append(('rsi', -1.0))
            elif rsi < 40: signals.append(('rsi', -0.5))
            else: signals.append(('rsi', 0))
            
        if 'macd' in data['indicators']:
            macd_data = data['indicators']['macd'].iloc[0]
            if macd_data['value'] > macd_data['signal']:
                signals.append(('macd_cross', 0.5))
            else:
                signals.append(('macd_cross', -0.5))
            
            if macd_data['histogram'] > 0:
                if macd_data['histogram'] > data['indicators']['macd']['histogram'].iloc[1]:
                    signals.append(('macd_hist', 1.0))
                else:
                    signals.append(('macd_hist', 0.5))
            else:
                if macd_data['histogram'] < data['indicators']['macd']['histogram'].iloc[1]:
                    signals.append(('macd_hist', -1.0))
                else:
                    signals.append(('macd_hist', -0.5))
        
        if 'ema' in data['indicators'] and 'sma' in data['indicators']:
            ema_values = data['indicators']['ema']['value']
            sma_values = data['indicators']['sma']['value']
            if len(ema_values) >= 3 and len(sma_values) >= 3:
                ema_trend = (ema_values.iloc[0] - ema_values.iloc[2]) / ema_values.iloc[2] * 100
                sma_trend = (sma_values.iloc[0] - sma_values.iloc[2]) / sma_values.iloc[2] * 100
                
                if ema_trend > 1: signals.append(('ema', 1.0))
                elif ema_trend > 0.2: signals.append(('ema', 0.5))
                elif ema_trend < -1: signals.append(('ema', -1.0))
                elif ema_trend < -0.2: signals.append(('ema', -0.5))
                else: signals.append(('ema', 0))
                
                if sma_trend > 1: signals.append(('sma', 1.0))
                elif sma_trend > 0.2: signals.append(('sma', 0.5))
                elif sma_trend < -1: signals.append(('sma', -1.0))
                elif sma_trend < -0.2: signals.append(('sma', -0.5))
                else: signals.append(('sma', 0))
                
                if ema_values.iloc[0] > sma_values.iloc[0] and ema_values.iloc[1] <= sma_values.iloc[1]:
                    signals.append(('ma_cross', 1.0))
                elif ema_values.iloc[0] < sma_values.iloc[0] and ema_values.iloc[1] >= sma_values.iloc[1]:
                    signals.append(('ma_cross', -1.0))
                elif ema_values.iloc[0] > sma_values.iloc[0]:
                    signals.append(('ma_cross', 0.5))
                else:
                    signals.append(('ma_cross', -0.5))
        
        confidence_scores = analyzer.calculate_confidence(signals, data['indicators'], timeframe)
        weights = {
            'signal_agreement': 0.4,
            'signal_strength': 0.3,
            'trend_consistency': 0.3
        }
        
        # Calculate individual scores
        signal_values = [s[1] for s in signals]
        if signal_values:
            max_diff = max(signal_values) - min(signal_values)
            signal_agreement = (1 - max_diff/2) * 100
            
            abs_strengths = [abs(v) for v in signal_values]
            avg_strength = sum(abs_strengths) / len(abs_strengths)
            signal_strength = avg_strength * 100
            
            trend_score = 0
            count = 0
            
            if 'ema' in data['indicators'] and not data['indicators']['ema'].empty:
                ema_values = data['indicators']['ema']['value']
                if len(ema_values) >= 3:
                    moves = [1 if ema_values.iloc[i] > ema_values.iloc[i+1] else -1 
                            for i in range(len(ema_values)-1)]
                    consistency = sum(1 for i in range(len(moves)-1) 
                                    if moves[i] == moves[i+1]) / (len(moves)-1)
                    trend_score += consistency * 100
                    count += 1
            
            if 'rsi' in data['indicators'] and not data['indicators']['rsi'].empty:
                rsi_values = data['indicators']['rsi']['value']
                if len(rsi_values) >= 3:
                    moves = [1 if rsi_values.iloc[i] > rsi_values.iloc[i+1] else -1 
                            for i in range(len(rsi_values)-1)]
                    consistency = sum(1 for i in range(len(moves)-1) 
                                    if moves[i] == moves[i+1]) / (len(moves)-1)
                    trend_score += consistency * 100
                    count += 1
            
            if 'sma' in data['indicators'] and not data['indicators']['sma'].empty:
                sma_values = data['indicators']['sma']['value']
                if len(sma_values) >= 3:
                    moves = [1 if sma_values.iloc[i] > sma_values.iloc[i+1] else -1 
                            for i in range(len(sma_values)-1)]
                    consistency = sum(1 for i in range(len(moves)-1) 
                                    if moves[i] == moves[i+1]) / (len(moves)-1)
                    trend_score += consistency * 100
                    count += 1
            
            trend_consistency = trend_score / max(1, count)
            
            print(f"Signal Agreement (40%): {signal_agreement:.1f}%")
            print(f"Signal Strength (30%): {signal_strength:.1f}%")
            print(f"Trend Consistency (30%): {trend_consistency:.1f}%")
            
            weighted_scores = {
                'Signal Agreement': signal_agreement * weights['signal_agreement'],
                'Signal Strength': signal_strength * weights['signal_strength'],
                'Trend Consistency': trend_consistency * weights['trend_consistency']
            }
            print("\nWeighted Contributions:")
            for name, score in weighted_scores.items():
                print(f"{name}: {score:.1f}")
    
    # Print market structure
    print("\nOverall Market Structure:")
    print("=" * 50)
    structure = analyzer.get_market_structure(results)
    print(f"Primary Trend (4h): {structure['primary_trend'].value if structure['primary_trend'] else 'N/A'}")
    print(f"Secondary Trend (1h): {structure['secondary_trend'].value if structure['secondary_trend'] else 'N/A'}")
    print(f"Entry Trend (15m): {structure['entry_trend'].value if structure['entry_trend'] else 'N/A'}")
    print(f"Confluence Level: {structure['confluence_level']:.1f}%")
    print(f"Trading Bias: {structure['trading_bias'].value}")
    print(f"Overall Confidence: {structure['confidence']:.1f}%")

if __name__ == "__main__":
    main()
