from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Tuple, Optional

from ai_st_data import AISTDataFetcher

class TimeframeAnalyzer:
    """Analyze market data across multiple timeframes."""
    
    def __init__(self):
        self.fetcher = AISTDataFetcher()
        
    def _calculate_volatility(self, high: float, low: float) -> Tuple[str, float]:
        """Calculate volatility level and percentage."""
        volatility = (high - low) / low * 100
        
        if volatility < 0.5:
            return "low", volatility
        elif volatility < 2.0:
            return "medium", volatility
        else:
            return "high", volatility
            
    def _get_trend_strength(self, rsi: float, macd_hist: float, ema_align: str) -> str:
        """Calculate trend strength based on indicators."""
        strength = 0
        
        # RSI contribution
        if 30 <= rsi <= 70:
            strength += 0
        elif (rsi < 30 and macd_hist > 0) or (rsi > 70 and macd_hist < 0):
            strength += 1
        else:
            strength -= 1
            
        # MACD contribution
        if abs(macd_hist) > 0.01:
            strength += 1 if macd_hist > 0 else -1
            
        # EMA contribution
        if ema_align == "bullish":
            strength += 1
        elif ema_align == "bearish":
            strength -= 1
            
        if strength >= 2:
            return "strong_bullish"
        elif strength == 1:
            return "bullish"
        elif strength == 0:
            return "neutral"
        elif strength == -1:
            return "bearish"
        else:
            return "strong_bearish"
            
    def _get_confidence_level(self, aligned_indicators: List[str]) -> str:
        """Calculate confidence level based on aligned indicators."""
        if len(aligned_indicators) >= 3:
            return "high"
        elif len(aligned_indicators) == 2:
            return "medium"
        else:
            return "low"
            
    def _get_rsi_trend(self, rsi: float) -> str:
        """Determine RSI trend."""
        if rsi < 30:
            return "strongly_oversold"
        elif rsi < 40:
            return "oversold"
        elif rsi > 70:
            return "strongly_overbought"
        elif rsi > 60:
            return "overbought"
        else:
            return "neutral"
            
    def _analyze_macd(self, macd_data: Dict) -> Dict:
        """Analyze MACD indicator with better trend detection."""
        if not isinstance(macd_data, dict):
            return {
                "trend": {
                    "value": "neutral",
                    "description": "Tendance du MACD"
                },
                "current_histogram": {
                    "value": 0,
                    "description": "Valeur actuelle de l'histogramme MACD"
                },
                "evolution_histogram": {
                    "value": 0,
                    "description": "Evolution de l'histogramme du MACD"
                },
                "recent_histogram": {
                    "values": [0] * 5,
                    "description": "Valeurs recentes de l'histogramme MACD"
                }
            }

        # Get histogram values
        histogram_data = macd_data.get('histogram', pd.Series([]))
        if isinstance(histogram_data, pd.Series) and not histogram_data.empty:
            recent_values = histogram_data.iloc[-5:].tolist() if len(histogram_data) >= 5 else [histogram_data.iloc[-1]] * 5
            current_value = histogram_data.iloc[-1]
            evolution = current_value - histogram_data.iloc[-2] if len(histogram_data) > 1 else 0
        else:
            recent_values = [0] * 5
            current_value = 0
            evolution = 0

        # Determine trend
        if current_value > 0:
            trend = "bullish"
            strength = "strong" if evolution > 0 else "weak"
        elif current_value < 0:
            trend = "bearish"
            strength = "strong" if evolution < 0 else "weak"
        else:
            trend = "neutral"
            strength = "weak"

        return {
            "trend": {
                "value": trend,
                "description": "Tendance du MACD"
            },
            "current_histogram": {
                "value": round(current_value, 5),
                "description": "Valeur actuelle de l'histogramme MACD"
            },
            "evolution_histogram": {
                "value": round(evolution, 5),
                "description": "Evolution de l'histogramme du MACD"
            },
            "recent_histogram": {
                "values": [round(x, 5) for x in recent_values],
                "description": "Valeurs recentes de l'histogramme MACD"
            },
            "strength": strength
        }

    def analyze_timeframe(self, ticker: str, timeframe: str) -> dict:
        """Analyze a specific timeframe."""
        config = {
            "5m": {"timespan": "minute", "multiplier": 5},
            "15m": {"timespan": "minute", "multiplier": 15},
            "1h": {"timespan": "hour", "multiplier": 1},
            "4h": {"timespan": "hour", "multiplier": 4},
            "1d": {"timespan": "day", "multiplier": 1}
        }[timeframe]
        
        price_data, indicators = self.fetcher.get_market_data(ticker, config['timespan'], config['multiplier'])
        
        if price_data.empty or not indicators:
            return None
            
        latest_price = price_data['close'].iloc[-1]
        high = price_data['high'].max()
        low = price_data['low'].min()
        
        volatility_level, volatility_pct = self._calculate_volatility(high, low)
        
        # Get RSI and trend
        rsi_data = indicators.get('rsi', {}).get('value')
        if isinstance(rsi_data, pd.Series) and not rsi_data.empty:
            rsi = rsi_data.iloc[-1]
            rsi_values = rsi_data.iloc[-5:].tolist()  # Prendre les 5 dernières valeurs
            rsi_evolution = rsi - rsi_data.iloc[-2] if len(rsi_data) > 1 else 0
        else:
            rsi = 50  # Valeur neutre par défaut
            rsi_values = [50] * 5
            rsi_evolution = 0
        
        rsi_trend = self._get_rsi_trend(rsi)
        
        # Get MACD with improved analysis
        macd_analysis = self._analyze_macd(indicators.get('macd', {}))
        
        # Get EMAs and alignment
        ema20_data = indicators.get('ema20', {}).get('value')
        ema50_data = indicators.get('ema50', {}).get('value')
        
        ema_20 = ema20_data.iloc[-1] if isinstance(ema20_data, pd.Series) and not ema20_data.empty else latest_price
        ema_50 = ema50_data.iloc[-1] if isinstance(ema50_data, pd.Series) and not ema50_data.empty else latest_price
        ema_align = "bullish" if ema_20 > ema_50 else "bearish"

        # Get SMAs
        sma20_data = indicators.get('sma20', {}).get('value')
        sma50_data = indicators.get('sma50', {}).get('value')
        
        sma_20 = sma20_data.iloc[-1] if isinstance(sma20_data, pd.Series) and not sma20_data.empty else latest_price
        sma_50 = sma50_data.iloc[-1] if isinstance(sma50_data, pd.Series) and not sma50_data.empty else latest_price
        
        trend = self._get_trend_strength(rsi, macd_analysis["current_histogram"]["value"], ema_align)
        
        # Récupérer et formater les 5 derniers prix de clôture avec leurs timestamps
        historical_data = price_data['close'].iloc[-5:]
        historical_prices = [
            {
                "timestamp": index.strftime("%Y-%m-%dT%H:%M:%S"),
                "value": round(float(value), 6)
            } for index, value in historical_data.items()
        ]
        
        # S'assurer que latest_price est le dernier prix
        if historical_prices:
            historical_prices[-1]["value"] = round(float(latest_price), 6)

        # Calcul des données de volume
        current_volume = float(price_data['volume'].iloc[-1])
        avg_volume = float(price_data['volume'].rolling(window=5).mean().iloc[-1])
        volume_trend = "increasing" if current_volume > avg_volume else "decreasing"
        
        # Récupération du VWAP
        current_vwap = float(price_data['vwap'].iloc[-1]) if 'vwap' in price_data else None

        # Récupérer les valeurs RSI avec timestamps
        rsi_data = indicators.get('rsi', {}).get('value')
        rsi_values = [
            {
                "timestamp": index.strftime("%Y-%m-%dT%H:%M:%S"),
                "value": round(float(value), 2)
            } for index, value in rsi_data.iloc[-5:].items()
        ]

        # Récupérer les valeurs MACD avec timestamps
        macd_hist_data = macd_analysis["recent_histogram"]["values"]
        macd_hist_timestamps = price_data.index[-5:]
        macd_hist_values = [
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "value": round(float(val), 5)
            } for ts, val in zip(macd_hist_timestamps, macd_hist_data)
        ]

        return {
            "timeframe": timeframe,
            "price": {
                "current": {
                    "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                    "value": round(float(latest_price), 6),
                    "description": "Prix actuel de l'actif"
                },
                "historical": {
                    "data": historical_prices,
                    "description": "Prix historique sur les dernieres periodes (du plus ancien au plus recent)"
                }
            },
            "volume": {
                "current": round(current_volume, 2),
                "average": round(avg_volume, 2),
                "trend": volume_trend,
                "vwap": round(current_vwap, 6) if current_vwap else None
            },
            "trend": trend,
            "indicators": {
                "RSI": {
                    "current": {
                        "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                        "value": round(rsi, 2),
                        "description": "Relative Strength Index (RSI) actuel"
                    },
                    "evolution": {
                        "value": round(rsi_evolution, 2),
                        "description": "Evolution du RSI par rapport a la periode precedente"
                    },
                    "recent_values": {
                        "data": rsi_values,
                        "description": "Valeurs recentes du RSI sur differentes periodes"
                    },
                    "trend": rsi_trend
                },
                "MACD": {
                    "trend": macd_analysis["trend"],
                    "current_histogram": {
                        "timestamp": price_data.index[-1].strftime("%Y-%m-%dT%H:%M:%S"),
                        "value": macd_analysis["current_histogram"]["value"],
                        "description": "Valeur actuelle de l'histogramme MACD"
                    },
                    "evolution_histogram": {
                        "value": macd_analysis["evolution_histogram"]["value"],
                        "description": "Evolution de l'histogramme du MACD"
                    },
                    "recent_histogram": {
                        "data": macd_hist_values,
                        "description": "Valeurs recentes de l'histogramme MACD"
                    },
                    "strength": macd_analysis["strength"]
                },
                "EMA": {
                    "ema_20": {
                        "value": round(ema_20, 2),
                        "description": "Exponential Moving Average (EMA) sur 20 periodes"
                    },
                    "ema_50": {
                        "value": round(ema_50, 2),
                        "description": "Exponential Moving Average (EMA) sur 50 periodes"
                    },
                    "trend": ema_align,
                    "description": f"Tendance de l'EMA : {'haussiere' if ema_align == 'bullish' else 'baissiere'}"
                },
                "SMA": {
                    "sma_20": {
                        "value": round(sma_20, 2),
                        "description": "Simple Moving Average (SMA) sur 20 periodes"
                    },
                    "sma_50": {
                        "value": round(sma_50, 2),
                        "description": "Simple Moving Average (SMA) sur 50 periodes"
                    }
                }
            },
            "volatility": volatility_level
        }

    def _identify_confluence_zones(self, timeframes_data: Dict) -> Dict:
        """Identify high and low confidence confluence zones."""
        zones = {
            "high_confidence": [],
            "low_confidence": []
        }
        
        # Group similar price levels across timeframes
        price_levels = []
        for tf, data in timeframes_data.items():
            if data is None:
                continue
                
            price = data["price"]["current"]["value"]
            ema20 = data["indicators"]["EMA"]["ema_20"]["value"]
            ema50 = data["indicators"]["EMA"]["ema_50"]["value"]
            
            price_levels.extend([
                (price, f"Price ({tf})", data["indicators"]),
                (ema20, f"EMA20 ({tf})", data["indicators"]),
                (ema50, f"EMA50 ({tf})", data["indicators"])
            ])
            
        # Group close price levels (within 0.5%)
        grouped_levels = {}
        for price, label, indicators in sorted(price_levels):
            found_group = False
            for base_price in grouped_levels.keys():
                if abs(price - base_price) / base_price < 0.005:
                    grouped_levels[base_price].append((label, indicators))
                    found_group = True
                    break
            if not found_group:
                grouped_levels[price] = [(label, indicators)]
                
        # Classify zones based on indicator alignment
        for price, alignments in grouped_levels.items():
            if len(alignments) < 2:
                continue
                
            aligned_indicators = []
            for label, indicators in alignments:
                rsi = indicators["RSI"]
                macd = indicators["MACD"]
                ema = indicators["EMA"]
                
                if rsi["trend"].endswith("oversold"):
                    aligned_indicators.append(f"{label} RSI oversold")
                elif rsi["trend"].endswith("overbought"):
                    aligned_indicators.append(f"{label} RSI overbought")
                    
                if macd["trend"]["value"] == "bullish":
                    aligned_indicators.append(f"{label} MACD bullish")
                elif macd["trend"]["value"] == "bearish":
                    aligned_indicators.append(f"{label} MACD bearish")
                    
                if ema["trend"] == "bullish":
                    aligned_indicators.append(f"{label} EMA bullish")
                elif ema["trend"] == "bearish":
                    aligned_indicators.append(f"{label} EMA bearish")
                    
            zone = {
                "range": [round(price * 0.998, 2), round(price * 1.002, 2)],
                "aligned_indicators": aligned_indicators
            }
            
            if len(aligned_indicators) >= 3:
                zones["high_confidence"].append(zone)
            else:
                zones["low_confidence"].append(zone)
                
        return zones
        
    def _generate_trade_suggestion(self, timeframes_data: Dict, confluence_zones: Dict) -> Dict:
        """Generate trade suggestion based on analysis."""
        # Check if we have enough data
        if not timeframes_data or not confluence_zones:
            return {
                "action": "Wait",
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "reason": "Insufficient data for trade suggestion"
            }
            
        # Look for strong setups in high confidence zones
        high_conf_zones = confluence_zones.get("high_confidence", [])
        if high_conf_zones:
            zone = high_conf_zones[0]  # Take the first high confidence zone
            
            # Determine if it's a buy or sell setup
            bullish_signals = sum(1 for i in zone["aligned_indicators"] if "bullish" in i.lower())
            bearish_signals = sum(1 for i in zone["aligned_indicators"] if "bearish" in i.lower())
            
            if abs(bullish_signals - bearish_signals) >= 2:  # Strong directional bias
                action = "Buy" if bullish_signals > bearish_signals else "Sell"
                entry = sum(zone["range"]) / 2  # Middle of the zone
                
                # Calculate stop loss and take profit
                volatility = max(v["volatility"] for v in timeframes_data.values() if v)
                risk_pct = 0.02 if volatility == "low" else 0.03 if volatility == "medium" else 0.05
                
                if action == "Buy":
                    stop_loss = round(entry * (1 - risk_pct), 4)
                    take_profit = round(entry * (1 + risk_pct * 3), 4)  # 3:1 reward ratio
                else:
                    stop_loss = round(entry * (1 + risk_pct), 4)
                    take_profit = round(entry * (1 - risk_pct * 3), 4)
                    
                return {
                    "action": action,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": f"Strong {action.lower()} setup in high confidence zone with multiple aligned indicators"
                }
                
        # No clear setup found
        return {
            "action": "Wait",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": "No clear confluence across timeframes; wait for confirmation"
        }
        
    def analyze_all(self, symbol: str) -> dict:
        """Analyze all timeframes and generate complete analysis."""
        timeframes = ["5m", "15m", "1h", "4h", "1d"]  
        timeframes_data = {}
        
        # Analyze each timeframe
        for timeframe in timeframes:
            timeframes_data[timeframe] = self.analyze_timeframe(symbol, timeframe)
            
        # Skip if we don't have enough data
        if not any(timeframes_data.values()):
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                "error": "Insufficient data for analysis"
            }
            
        # Identify confluence zones
        confluence_zones = self._identify_confluence_zones(timeframes_data)
        
        # Generate trade suggestion
        trade_suggestion = self._generate_trade_suggestion(timeframes_data, confluence_zones)
        
        # Calculate overall market bias and confidence
        trends = [data["trend"] for data in timeframes_data.values() if data]
        bias = max(set(trends), key=trends.count) if trends else "neutral"
        
        # Calculate confidence level
        confidence = self._get_confidence_level(
            sum((z.get("aligned_indicators", []) for z in confluence_zones.get("high_confidence", [])), [])
        )
        
        # Calculate overall volatility
        volatilities = [data["volatility"] for data in timeframes_data.values() if data]
        overall_volatility = max(set(volatilities), key=volatilities.count) if volatilities else "low"
        
        # Determine trade readiness
        if trade_suggestion["action"] == "Wait":
            trade_readiness = "Awaiting confirmation"
        else:
            trade_readiness = f"Ready to {trade_suggestion['action'].lower()}"
            
        # Find critical levels from confluence zones
        critical_levels = {
            "buy_zone": None,
            "sell_zone": None
        }
        
        if confluence_zones["high_confidence"]:
            for zone in confluence_zones["high_confidence"]:
                if any("bullish" in i.lower() for i in zone["aligned_indicators"]):
                    critical_levels["buy_zone"] = zone["range"]
                if any("bearish" in i.lower() for i in zone["aligned_indicators"]):
                    critical_levels["sell_zone"] = zone["range"]
                    
        # Format the final output
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "bias": bias,
                "confidence_level": confidence,
                "volatility": overall_volatility,
                "trade_readiness": trade_readiness,
                "critical_levels": critical_levels
            },
            "timeframes": timeframes_data,
            "confluence_zones": {
                "high_confidence": next(iter(confluence_zones["high_confidence"]), {}) if confluence_zones["high_confidence"] else {}
            },
            "suggested_trade": {
                "action": trade_suggestion["action"],
                "entry": trade_suggestion["entry"],
                "stop_loss": trade_suggestion["stop_loss"],
                "take_profit": trade_suggestion["take_profit"],
                "risk_reward": round(abs(trade_suggestion["take_profit"] - trade_suggestion["entry"]) / 
                                  abs(trade_suggestion["stop_loss"] - trade_suggestion["entry"]), 1) 
                                  if all(x is not None for x in [trade_suggestion["entry"], 
                                                               trade_suggestion["stop_loss"], 
                                                               trade_suggestion["take_profit"]]) else None
            }
        }

def main():
    """Main function to run the analyzer."""
    if len(sys.argv) != 2:
        print("Usage: python timeframe_analyzer.py <symbol>")
        sys.exit(1)
        
    symbol = sys.argv[1]
    # Add /USD only if not already present
    if not symbol.endswith("/USD"):
        symbol = f"{symbol}/USD"
        
    analyzer = TimeframeAnalyzer()
    analysis = analyzer.analyze_all(symbol)
    
    # Pretty print the analysis
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()
