# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict

# Third-party imports
import operator
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START

# Local imports
from src.polygon_tools import (
    get_crypto_metrics as get_polygon_metrics,
    get_price_data,
    get_technical_indicators
)
from src.coingecko_tools import CoinGeckoAPI
from src.fear_greed_tools import FearGreedIndex

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model="gpt-4")
coingecko = CoinGeckoAPI()
fear_greed = FearGreedIndex()

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries."""
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]


def market_data_agent(state: dict) -> dict:
    """Gather and analyze market data from multiple sources."""
    data = state["data"]
    messages = state["messages"]
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=data.get("lookback_days", 30))).strftime("%Y-%m-%d")

    try:
        # 1. Technical Analysis (Polygon)
        technical_data = get_technical_indicators(data["ticker"], start_date)
        
        # 2. Market Metrics (Polygon)
        polygon_data = get_polygon_metrics(data["ticker"])
        
        # 3. Additional Metrics (CoinGecko)
        coingecko_data = coingecko.get_key_metrics(data["ticker"])
        
        # 4. Market Sentiment (Fear & Greed Index)
        sentiment_data = fear_greed.get_current_index()
        sentiment_history = fear_greed.get_historical_index(7)  # Last 7 days
        market_sentiment = fear_greed.get_market_sentiment()
        
        # Combine all data sources
        data.update({
            "technical_analysis": {
                "indicators": technical_data,
                "trends": {
                    "macd_trend": "bullish" if technical_data['macd']['histogram'] > 0 else "bearish",
                    "rsi_signal": "overbought" if technical_data['rsi'] > 70 else "oversold" if technical_data['rsi'] < 30 else "neutral",
                    "ma_trend": "bullish" if technical_data['sma'] < technical_data['ema'] else "bearish"
                }
            },
            "market_metrics": {
                "polygon": {
                    "market_cap": polygon_data.get("market_cap", 0),
                    "volume_24h": polygon_data.get("volume_24h", 0),
                    "vwap_24h": polygon_data.get("vwap_24h", 0),
                },
                "coingecko": {
                    "price_usd": coingecko_data.get("price_usd", 0),
                    "market_cap": coingecko_data.get("market_cap_usd", 0),
                    "volume_24h": coingecko_data.get("volume_24h_usd", 0),
                    "price_change_24h": coingecko_data.get("change_24h", 0),
                    "price_change_7d": coingecko_data.get("change_7d", 0),
                    "volatility_24h": coingecko_data.get("volatility_24h", 0)
                }
            },
            "fundamental_metrics": {
                "supply_metrics": {
                    "circulating_supply": polygon_data.get("circulating_supply", 0),
                    "max_supply": polygon_data.get("max_supply", 0),
                    "supply_ratio": coingecko_data.get("supply_ratio", 0)
                },
                "development_metrics": {
                    "github_commits": coingecko_data.get("dev_commits_4w", 0),
                    "github_stars": coingecko_data.get("dev_stars", 0)
                },
                "social_metrics": {
                    "twitter_followers": coingecko_data.get("twitter_followers", 0),
                    "sentiment_score": coingecko_data.get("sentiment_votes_up_percentage", 0)
                }
            },
            "market_sentiment": {
                "fear_greed_value": sentiment_data.get("value", 0),
                "fear_greed_classification": sentiment_data.get("value_classification", "unknown"),
                "market_sentiment": market_sentiment,
                "sentiment_trend": [
                    {
                        "date": entry["timestamp"],
                        "value": entry["value"],
                        "classification": entry["value_classification"]
                    }
                    for entry in (sentiment_history or [])
                ]
            }
        })

        # Create comprehensive market analysis message
        messages.append(
            HumanMessage(
                content=f"Market Analysis for {data['ticker']}:\n\n"
                f"1. Technical Analysis:\n"
                f"- MACD: {technical_data['macd']['histogram']:.2f} ({data['technical_analysis']['trends']['macd_trend']})\n"
                f"- RSI: {technical_data['rsi']:.2f} ({data['technical_analysis']['trends']['rsi_signal']})\n"
                f"- Moving Averages: {data['technical_analysis']['trends']['ma_trend']}\n\n"
                
                f"2. Market Metrics:\n"
                f"- Price: ${coingecko_data['price_usd']:,.2f}\n"
                f"- 24h Change: {coingecko_data['change_24h']:+.2f}%\n"
                f"- 7d Change: {coingecko_data['change_7d']:+.2f}%\n"
                f"- 24h Volatility: {coingecko_data['volatility_24h']:.2f}%\n"
                f"- Market Cap: ${coingecko_data['market_cap_usd']:,.2f}\n"
                f"- 24h Volume: ${coingecko_data['volume_24h_usd']:,.2f}\n\n"
                
                f"3. Development & Social Metrics:\n"
                f"- Recent GitHub Commits: {coingecko_data['dev_commits_4w']}\n"
                f"- GitHub Stars: {coingecko_data['dev_stars']:,}\n"
                f"- Twitter Followers: {coingecko_data['twitter_followers']:,}\n"
                f"- Community Sentiment: {coingecko_data['sentiment_votes_up_percentage']:.1f}% Positive\n\n"
                
                f"4. Market Sentiment:\n"
                f"- Fear & Greed Index: {sentiment_data['value']}/100\n"
                f"- Classification: {sentiment_data['value_classification']}\n"
                f"- Overall Market Sentiment: {market_sentiment.upper()}\n"
            )
        )

    except Exception as e:
        messages.append(
            HumanMessage(content=f"Error in market data agent: {str(e)}")
        )

    return {"messages": messages, "data": data}


def quant_agent(state: AgentState) -> AgentState:
    """Quantitative analysis agent for cryptocurrency trading."""
    data = state["data"]
    messages = state["messages"]

    try:
        # Technical indicators are already calculated in market_data_agent
        tech_analysis = data["technical_analysis"]
        fund_analysis = data["fundamental_metrics"]

        # Combine analyses to make decision
        macd_signal = tech_analysis["indicators"]["macd"]["macd_line"]
        rsi_signal = tech_analysis["indicators"]["rsi"]

        # Trading decision logic
        decision = "hold"  # Default decision
        quantity = 0
        confidence = 0.5  # Base confidence

        # RSI-based decision
        if rsi_signal < 30:  # Oversold
            decision = "buy"
            confidence += 0.2
        elif rsi_signal > 70:  # Overbought
            decision = "sell"
            confidence += 0.2

        # MACD-based decision
        if macd_signal > 0 and tech_analysis["indicators"]["macd"]["histogram"] > 0:  # Bullish
            if decision == "buy":
                confidence += 0.2
            elif decision == "hold":
                decision = "buy"
                confidence += 0.1
        elif macd_signal < 0 and tech_analysis["indicators"]["macd"]["histogram"] < 0:  # Bearish
            if decision == "sell":
                confidence += 0.2
            elif decision == "hold":
                decision = "sell"
                confidence += 0.1

        # Market strength consideration
        if fund_analysis["supply_metrics"]["market_cap"] > 10000000000:
            if decision == "buy":
                confidence += 0.1
        else:
            if decision == "sell":
                confidence += 0.1

        # Calculate quantity based on confidence
        if decision != "hold":
            portfolio = data.get("portfolio", {"cash": 10000, "assets": 0})
            if decision == "buy":
                max_quantity = portfolio["cash"] / tech_analysis["indicators"]["ema"]  # Use EMA as current price
                quantity = max_quantity * confidence
            else:  # sell
                quantity = portfolio["assets"] * confidence

        # Round quantity to 8 decimal places (common in crypto)
        quantity = round(quantity, 8)

        messages.append(
            HumanMessage(
                content=f"Trading Decision for {data['ticker']}:\n"
                f"Action: {decision.upper()}\n"
                f"Quantity: {quantity}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Based on:\n"
                f"- RSI: {rsi_signal:.2f}\n"
                f"- MACD Signal: {macd_signal:.2f}\n"
                f"- Market Cap: ${fund_analysis['supply_metrics']['market_cap']:,.2f}"
            )
        )

    except Exception as e:
        messages.append(
            HumanMessage(
                content=f"Error in technical analysis: {str(e)}"
            )
        )

    return {"messages": messages, "data": data}


def fundamentals_agent(state: AgentState) -> AgentState:
    """Fundamental analysis agent for cryptocurrency markets."""
    data = state["data"]
    messages = state["messages"]

    try:
        metrics = data.get("fundamental_metrics", {})
        
        analysis = {
            "market_strength": "strong" if metrics["supply_metrics"]["market_cap"] > 10000000000 else "weak",
            "supply_analysis": {
                "circulating_supply": metrics["supply_metrics"].get("circulating_supply", 0),
                "max_supply": metrics["supply_metrics"].get("max_supply", 0),
                "supply_ratio": metrics["supply_metrics"].get("circulating_supply", 0) / metrics["supply_metrics"].get("max_supply", 1) if metrics["supply_metrics"].get("max_supply", 0) else 1
            },
            "market_cap_category": "large" if metrics["supply_metrics"]["market_cap"] > 10000000000 else 
                                 "medium" if metrics["supply_metrics"]["market_cap"] > 1000000000 else "small"
        }
        
        data["fundamental_metrics"] = analysis
        
        messages.append(
            HumanMessage(
                content=f"Fundamental Analysis for {data['ticker']}:\n"
                f"Market Strength: {analysis['market_strength']}\n"
                f"Supply Ratio: {analysis['supply_analysis']['supply_ratio']:.2%}\n"
                f"Market Cap Category: {analysis['market_cap_category']}"
            )
        )

    except Exception as e:
        messages.append(
            HumanMessage(
                content=f"Error in fundamental analysis: {str(e)}"
            )
        )

    return {"messages": messages, "data": data}


def decision_agent(state: dict) -> dict:
    """Make trading decisions based on technical and fundamental analysis."""
    data = state["data"]
    messages = state["messages"]

    try:
        # Technical indicators are already calculated in market_data_agent
        tech_analysis = data["technical_analysis"]
        fund_analysis = data["fundamental_metrics"]

        # Combine analyses to make decision
        macd = tech_analysis["indicators"]["macd"]
        rsi_signal = tech_analysis["indicators"]["rsi"]

        # Trading decision logic
        decision = "hold"  # Default decision
        quantity = 0
        confidence = 0.5  # Base confidence

        # RSI-based decision
        if rsi_signal < 30:  # Oversold
            decision = "buy"
            confidence += 0.2
        elif rsi_signal > 70:  # Overbought
            decision = "sell"
            confidence += 0.2

        # MACD-based decision
        if macd["macd_line"] > macd["signal_line"] and macd["histogram"] > 0:  # Bullish
            if decision == "buy":
                confidence += 0.2
            elif decision == "hold":
                decision = "buy"
                confidence += 0.1
        elif macd["macd_line"] < macd["signal_line"] and macd["histogram"] < 0:  # Bearish
            if decision == "sell":
                confidence += 0.2
            elif decision == "hold":
                decision = "sell"
                confidence += 0.1

        # Market strength consideration
        if fund_analysis["supply_metrics"]["market_cap"] > 10000000000:  # 10B market cap threshold
            if decision == "buy":
                confidence += 0.1
        else:
            if decision == "sell":
                confidence += 0.1

        # Calculate quantity based on confidence
        if decision != "hold":
            portfolio = data.get("portfolio", {"cash": 10000, "assets": 0})
            if decision == "buy":
                max_quantity = portfolio["cash"] / tech_analysis["indicators"]["ema"]  # Use EMA as current price
                quantity = max_quantity * confidence
            else:  # sell
                quantity = portfolio["assets"] * confidence

        # Round quantity to 8 decimal places (common in crypto)
        quantity = round(quantity, 8)

        messages.append(
            HumanMessage(
                content=f"Trading Decision for {data['ticker']}:\n"
                f"Action: {decision.upper()}\n"
                f"Quantity: {quantity}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Based on:\n"
                f"- RSI: {rsi_signal:.2f}\n"
                f"- MACD Line: {macd['macd_line']:.2f}\n"
                f"- MACD Signal: {macd['signal_line']:.2f}\n"
                f"- MACD Histogram: {macd['histogram']:.2f}\n"
                f"- Market Cap: ${fund_analysis['supply_metrics']['market_cap']:,.2f}"
            )
        )

        # Update state with decision
        data.update({
            "decision": {
                "action": decision,
                "quantity": quantity,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })

    except Exception as e:
        messages.append(
            HumanMessage(content=f"Error in decision agent: {str(e)}")
        )

    return {"messages": messages, "data": data}


def display_results(final_state):
    """Display the results of the analysis in a readable format."""
    if final_state:
        print("\nAnalysis Results:")
        print("----------------")
        
        # Display technical analysis
        if "technical_analysis" in final_state.get("data", {}):
            tech = final_state["data"]["technical_analysis"]
            print("\nTechnical Analysis:")
            print("  MACD:")
            print(f"    - Line: {tech['indicators']['macd']['macd_line']:.2f}")
            print(f"    - Signal: {tech['indicators']['macd']['signal_line']:.2f}")
            print(f"    - Histogram: {tech['indicators']['macd']['histogram']:.2f}")
            print(f"    - Trend: {tech['trends']['macd_trend'].upper()}")
            print(f"  RSI: {tech['indicators']['rsi']:.2f} ({tech['trends']['rsi_signal'].upper()})")
            print(f"  Moving Averages: {tech['trends']['ma_trend'].upper()}")
        
        # Display market metrics
        if "market_metrics" in final_state.get("data", {}):
            metrics = final_state["data"]["market_metrics"]
            print("\nMarket Metrics:")
            print("  From CoinGecko:")
            print(f"    - Price: ${metrics['coingecko'].get('price_usd', 0):,.2f}")
            print(f"    - 24h Change: {metrics['coingecko'].get('price_change_24h', 0):+.2f}%")
            print(f"    - 7d Change: {metrics['coingecko'].get('price_change_7d', 0):+.2f}%")
            print(f"    - Market Cap: ${metrics['coingecko'].get('market_cap', 0):,.2f}")
            print(f"    - 24h Volume: ${metrics['coingecko'].get('volume_24h', 0):,.2f}")
            print(f"    - 24h Volatility: {metrics['coingecko'].get('volatility_24h', 0):.2f}%")
            print("  From Polygon:")
            print(f"    - Market Cap: ${metrics['polygon'].get('market_cap', 0):,.2f}")
            print(f"    - 24h Volume: ${metrics['polygon'].get('volume_24h', 0):,.2f}")
            print(f"    - 24h VWAP: ${metrics['polygon'].get('vwap_24h', 0):,.2f}")
        
        # Display fundamental metrics
        if "fundamental_metrics" in final_state.get("data", {}):
            fund = final_state["data"]["fundamental_metrics"]
            print("\nFundamental Analysis:")
            print("  Supply Metrics:")
            if fund['supply_metrics'].get('circulating_supply') and fund['supply_metrics'].get('max_supply'):
                supply_ratio = fund['supply_metrics']['circulating_supply'] / fund['supply_metrics']['max_supply'] if fund['supply_metrics']['max_supply'] > 0 else 0
                print(f"    - Circulating Supply: {fund['supply_metrics']['circulating_supply']:,.0f}")
                print(f"    - Maximum Supply: {fund['supply_metrics']['max_supply']:,.0f}")
                print(f"    - Supply Ratio: {supply_ratio:.2%}")
            print("  Development Metrics:")
            print(f"    - GitHub Commits (4w): {fund['development_metrics'].get('github_commits', 0):,}")
            print(f"    - GitHub Stars: {fund['development_metrics'].get('github_stars', 0):,}")
            print("  Social Metrics:")
            print(f"    - Twitter Followers: {fund['social_metrics'].get('twitter_followers', 0):,}")
            print(f"    - Community Sentiment: {fund['social_metrics'].get('sentiment_score', 0):.1f}% Positive")
        
        # Display market sentiment
        if "market_sentiment" in final_state.get("data", {}):
            sentiment = final_state["data"]["market_sentiment"]
            print("\nMarket Sentiment:")
            print(f"  Fear & Greed Index: {sentiment['fear_greed_value']}/100")
            print(f"  Classification: {sentiment['fear_greed_classification']}")
            print(f"  Overall Sentiment: {sentiment['market_sentiment']}")
            if sentiment.get('sentiment_trend'):
                print("  Recent Trend:")
                for entry in sentiment['sentiment_trend'][-5:]:  # Show last 5 days
                    date = datetime.strptime(entry['date'], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d")
                    print(f"    - {date}: {entry['value']}/100 ({entry['classification']})")
        
        # Display trading decision
        if "decision" in final_state.get("data", {}):
            decision = final_state["data"]["decision"]
            print("\nTrading Decision:")
            print(f"  Action: {decision['action'].upper()}")
            print(f"  Quantity: {decision['quantity']:.4f}")
            print(f"  Confidence: {decision['confidence']:.1%}")
            if 'timestamp' in decision:
                print(f"  Timestamp: {decision['timestamp']}")

def run_hedge_fund(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    portfolio: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the cryptocurrency hedge fund strategy.
    
    Args:
        ticker (str): Cryptocurrency ticker (e.g., 'BTC')
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        portfolio (dict, optional): Current portfolio state
        
    Returns:
        Dict[str, Any]: Trading decision and analysis
    """
    # Initialize state
    initial_state = {
        "messages": [],
        "data": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "portfolio": portfolio or {"cash": 10000, "assets": 0}
        },
        "metadata": {}
    }
    
    # Set up and run the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes for each analysis step
    workflow.add_node("market_data", market_data_agent)
    workflow.add_node("quant_analysis", quant_agent)
    workflow.add_node("fundamental_analysis", fundamentals_agent)
    workflow.add_node("decision", decision_agent)
    
    # Define the workflow sequence
    workflow.add_edge("market_data", "quant_analysis")
    workflow.add_edge("quant_analysis", "fundamental_analysis")
    workflow.add_edge("fundamental_analysis", "decision")
    workflow.add_edge("decision", END)
    
    # Set the entry point
    workflow.set_entry_point("market_data")
    
    # Run the workflow
    app = workflow.compile()
    final_state = app.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the cryptocurrency hedge fund trading system'
    )
    parser.add_argument(
        '--ticker',
        required=True,
        help='Cryptocurrency ticker (e.g., BTC)'
    )
    parser.add_argument(
        '--start_date',
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end_date',
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=10000.0,
        help='Initial capital for backtesting'
    )
    parser.add_argument(
        '--lookback_days',
        type=int,
        default=30,
        help='Number of days to look back for analysis'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode'
    )
    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show detailed reasoning for decisions'
    )

    args = parser.parse_args()
    
    if args.backtest:
        from src.backtester import Backtester
        
        # Set default dates for backtesting if not provided
        if not args.end_date:
            args.end_date = datetime.now().strftime("%Y-%m-%d")
        if not args.start_date:
            start_dt = datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=180)  # 6 months by default
            args.start_date = start_dt.strftime("%Y-%m-%d")
            
        # Create and run backtester
        backtester = Backtester(
            agent=run_hedge_fund,
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            lookback_days=args.lookback_days
        )
        
        # Run backtest and analyze results
        backtester.run_backtest()
        backtester.analyze_performance()
        
    else:
        # Run single analysis
        result = run_hedge_fund(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            portfolio={"cash": args.initial_capital, "assets": 0}
        )
        
        if args.show_reasoning:
            display_results(result)
        else:
            decision = result.get("data", {}).get("decision", {"action": "hold", "quantity": 0})
            print(f"\nTrading Decision for {args.ticker}:")
            print(f"Action: {decision['action'].upper()}")
            print(f"Quantity: {decision['quantity']:.4f}")
