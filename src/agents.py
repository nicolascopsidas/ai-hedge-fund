# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict

# Third-party imports
import operator
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph, START

# Local imports
from src.polygon_tools import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    get_financial_metrics,
    get_insider_trades,
    get_price_data,
)

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries."""
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

def market_data_agent(state: Dict) -> Dict:
    """Market data agent responsible for gathering and preprocessing market data."""
    data = state["data"]
    messages = state.get("messages", [])
    
    end_date = data["end_date"]
    
    # Use start_date from data if available, otherwise calculate it
    if "start_date" in data:
        start_date = data["start_date"]
    else:
        # Default to 30 days before end_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")

    # Get the historical price data
    prices_df = get_price_data(
        ticker=data["ticker"],
        start_date=start_date,
        end_date=end_date,
        asset_type='crypto'
    )
    
    if prices_df.empty:
        return {
            "messages": [
                HumanMessage(content=json.dumps({
                    "error": "No price data available",
                    "ticker": data["ticker"],
                    "start_date": start_date,
                    "end_date": end_date
                }))
            ],
            "data": data
        }

    # Calculate technical indicators
    macd_line, signal_line, macd_hist = calculate_macd(prices_df)
    rsi = calculate_rsi(prices_df)
    upper_band, lower_band = calculate_bollinger_bands(prices_df)
    obv = calculate_obv(prices_df)

    # Get financial metrics and insider trades
    try:
        financial_metrics = get_financial_metrics(
            ticker=data["ticker"],
            report_period=end_date,
            asset_type='crypto'
        )
    except Exception as e:
        messages.append(
            HumanMessage(content=f"Warning: Could not fetch financial metrics: {str(e)}")
        )
        financial_metrics = []

    try:
        insider_trades = get_insider_trades(
            ticker=data["ticker"],
            start_date=start_date,
            end_date=end_date,
            asset_type='crypto'
        )
    except Exception as e:
        messages.append(
            HumanMessage(content=f"Warning: Could not fetch insider trades: {str(e)}")
        )
        insider_trades = []

    # Create a message about insider trades
    insider_trades_message = "Found insider trading activity." if insider_trades else "No insider trading activity found."
    messages.append(HumanMessage(content=insider_trades_message))

    return {
        "messages": messages,
        "data": {
            **data,
            "prices_df": prices_df,
            "start_date": start_date,
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "insider_trades": insider_trades,
            "technical_indicators": {
                "macd": {"line": macd_line, "signal": signal_line, "histogram": macd_hist},
                "rsi": rsi,
                "bollinger_bands": {"upper": upper_band, "lower": lower_band},
                "obv": obv
            }
        }
    }

def quant_agent(state: AgentState) -> Dict:
    """Quantitative analysis agent responsible for technical analysis and trading signals."""
    data = state["data"]
    messages = state["messages"]
    
    if "technical_indicators" not in data:
        return {
            "messages": messages + [
                HumanMessage(content="Error: Technical indicators not available")
            ],
            "data": data
        }

    tech_indicators = data["technical_indicators"]
    
    # Analyze MACD
    macd_data = tech_indicators["macd"]
    macd_line = macd_data["line"]
    signal_line = macd_data["signal"]
    macd_hist = macd_data["histogram"]
    
    # Get latest values using iloc
    latest_macd = macd_line.iloc[-1] if not macd_line.empty else None
    latest_signal = signal_line.iloc[-1] if not signal_line.empty else None
    latest_hist = macd_hist.iloc[-1] if not macd_hist.empty else None
    
    # Analyze RSI
    rsi = tech_indicators["rsi"]
    latest_rsi = rsi.iloc[-1] if not rsi.empty else None
    
    # Analyze Bollinger Bands
    bb = tech_indicators["bollinger_bands"]
    latest_upper = bb["upper"].iloc[-1] if not bb["upper"].empty else None
    latest_lower = bb["lower"].iloc[-1] if not bb["lower"].empty else None
    
    # Analyze OBV
    obv = tech_indicators["obv"]
    latest_obv = obv.iloc[-1] if not obv.empty else None
    
    # Generate analysis message
    analysis = {
        "macd": {
            "value": latest_macd,
            "signal": latest_signal,
            "histogram": latest_hist,
            "interpretation": "bullish" if latest_hist > 0 else "bearish" if latest_hist < 0 else "neutral"
        },
        "rsi": {
            "value": latest_rsi,
            "interpretation": "overbought" if latest_rsi > 70 else "oversold" if latest_rsi < 30 else "neutral"
        },
        "bollinger_bands": {
            "upper": latest_upper,
            "lower": latest_lower
        },
        "obv": {
            "value": latest_obv
        }
    }
    
    messages.append(
        HumanMessage(content=f"Technical analysis completed for {data['ticker']}:\n{json.dumps(analysis, indent=2)}")
    )
    
    return {
        "messages": messages,
        "data": {
            **data,
            "technical_analysis": analysis
        }
    }

def fundamentals_agent(state: AgentState) -> Dict:
    """Fundamental analysis agent responsible for analyzing financial metrics and insider trades."""
    data = state["data"]
    messages = state["messages"]
    
    if "financial_metrics" not in data or "insider_trades" not in data:
        return {
            "messages": messages + [
                HumanMessage(content="Error: Financial metrics or insider trades not available")
            ],
            "data": data
        }
    
    financial_metrics = data["financial_metrics"]
    insider_trades = data["insider_trades"]
    
    # Analyze financial metrics
    metrics_analysis = {}
    if financial_metrics:
        metrics_analysis = {
            "summary": "Financial metrics analysis completed",
            "metrics": financial_metrics
        }
    
    # Analyze insider trades
    insider_analysis = {}
    if insider_trades:
        insider_analysis = {
            "summary": "Insider trading activity detected",
            "trades": insider_trades
        }
    
    # Generate analysis message
    analysis = {
        "financial_metrics": metrics_analysis,
        "insider_trades": insider_analysis
    }
    
    messages.append(
        HumanMessage(content=f"Fundamental analysis completed for {data['ticker']}:\n{json.dumps(analysis, indent=2)}")
    )
    
    return {
        "messages": messages,
        "data": {
            **data,
            "fundamental_analysis": analysis
        }
    }

def decision_agent(state: AgentState) -> Dict:
    """Decision making agent that combines all analyses to make trading decisions."""
    data = state["data"]
    messages = state["messages"]
    
    if not all(key in data for key in ["technical_analysis", "fundamental_analysis"]):
        return {
            "messages": messages + [
                HumanMessage(content="Error: Missing required analysis data")
            ],
            "data": data
        }
    
    tech_analysis = data["technical_analysis"]
    fund_analysis = data["fundamental_analysis"]
    
    # Combine analyses to make decision
    macd_signal = tech_analysis["macd"]["interpretation"]
    rsi_signal = tech_analysis["rsi"]["interpretation"]
    
    # Simple decision logic based on technical indicators
    decision = "hold"  # Default decision
    
    if macd_signal == "bullish" and rsi_signal == "oversold":
        decision = "buy"
    elif macd_signal == "bearish" and rsi_signal == "overbought":
        decision = "sell"
    
    # Add fundamental factors if available
    if fund_analysis.get("insider_trades", {}).get("trades"):
        messages.append(
            HumanMessage(content="Note: Considering insider trading activity in decision")
        )
    
    messages.append(
        HumanMessage(content=f"Trading decision for {data['ticker']}: {decision.upper()}")
    )
    
    return {
        "messages": messages,
        "data": {
            **data,
            "trading_decision": {
                "action": decision,
                "confidence": 0.7,  # Placeholder confidence score
                "factors": {
                    "technical": tech_analysis,
                    "fundamental": fund_analysis
                }
            }
        }
    }

def run_hedge_fund(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    portfolio: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the hedge fund simulation with the given parameters."""
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        # Default to 30 days before end_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
    
    # Set default portfolio if not provided
    if portfolio is None:
        portfolio = {
            "cash": 100000.00,
            "assets": {}
        }
    
    # Initialize workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("market_data", market_data_agent)
    workflow.add_node("quant", quant_agent)
    workflow.add_node("fundamentals", fundamentals_agent)
    workflow.add_node("decision", decision_agent)
    
    # Add edges with START point
    workflow.add_edge(START, "market_data")
    workflow.add_edge("market_data", "quant")
    workflow.add_edge("market_data", "fundamentals")
    workflow.add_edge("quant", "decision")
    workflow.add_edge("fundamentals", "decision")
    workflow.add_edge("decision", END)
    
    # Compile workflow
    app = workflow.compile()
    
    # Run workflow
    final_state = app.invoke(
        {
            "messages": [],
            "data": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "portfolio": portfolio
            },
            "metadata": {}
        }
    )
    
    return final_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True,
                      help='The ticker symbol to trade')
    parser.add_argument('--start-date', type=str,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str,
                      help='End date in YYYY-MM-DD format')
    parser.add_argument('--show-reasoning', action='store_true',
                      help='Show the reasoning behind each decision')
    
    args = parser.parse_args()
    
    # Initialize portfolio
    portfolio = {
        "cash": 100000.00,
        "assets": {}
    }
    
    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=portfolio,
    )

    if args.show_reasoning:
        print("\n=== Analysis Results for", args.ticker, "===\n")
        
        # Technical Analysis
        tech = result['data']['technical_analysis']
        print("Technical Analysis:")
        print(f"  MACD:")
        print(f"    Value: {tech['macd']['value']:.2f}")
        print(f"    Signal: {tech['macd']['signal']:.2f}")
        print(f"    Histogram: {tech['macd']['histogram']:.2f}")
        print(f"    Interpretation: {tech['macd']['interpretation']}")
        
        print(f"\n  RSI:")
        print(f"    Value: {tech['rsi']['value']:.2f}")
        print(f"    Interpretation: {tech['rsi']['interpretation']}")
        
        print(f"\n  Bollinger Bands:")
        print(f"    Upper: {tech['bollinger_bands']['upper']:.2f}")
        print(f"    Lower: {tech['bollinger_bands']['lower']:.2f}")
        
        print(f"\n  On-Balance Volume:")
        print(f"    Value: {tech['obv']['value']:.2f}")
        
        # Fundamental Analysis
        fund = result['data']['fundamental_analysis']
        print("\nFundamental Analysis:")
        metrics = fund['financial_metrics']['metrics'][0]
        print(f"  24h Volume: {metrics['24h_volume']:.2f}")
        print(f"  24h VWAP: {metrics['24h_vwap']:.2f}")
        print(f"  24h Open: {metrics['24h_open']:.2f}")
        print(f"  24h Close: {metrics['24h_close']:.2f}")
        print(f"  24h High: {metrics['24h_high']:.2f}")
        print(f"  24h Low: {metrics['24h_low']:.2f}")
        print(f"  24h Transactions: {metrics['24h_transactions']}")
        
        # Trading Decision
        decision = result['data']['trading_decision']
        print("\nTrading Decision:")
        print(f"  Action: {decision['action'].upper()}")
        print(f"  Confidence: {decision['confidence']:.2%}")
        
        print("\nPortfolio Status:")
        print(f"  Cash: ${portfolio['cash']:,.2f}")
        if portfolio['assets']:
            print("  Assets:")
            for asset, amount in portfolio['assets'].items():
                print(f"    {asset}: {amount}")
        else:
            print("  Assets: None")
    else:
        print(f"\nTrading Decision for {args.ticker}: {result['data']['trading_decision']['action'].upper()}")
