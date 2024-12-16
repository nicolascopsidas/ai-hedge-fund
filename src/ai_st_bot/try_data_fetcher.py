from ai_st_data import AISTDataFetcher
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def test_timeframe(fetcher, timeframe, multiplier):
    print(f"\n=== Testing {multiplier} {timeframe} timeframe ===")
    indicators = fetcher.get_indicators("BTC", timespan=timeframe, multiplier=multiplier)
    
    for indicator_name, df in indicators.items():
        print(f"\n{indicator_name.upper()}:")
        if not df.empty:
            print(df.head(3))
        else:
            print("No data available")
    print("\n" + "="*50)

def main():
    # Initialize fetcher
    fetcher = AISTDataFetcher()
    
    # Test different timeframes
    timeframes = [
        (5, "minute"),
        (15, "minute"),
        (1, "hour"),
        (4, "hour"),
        (1, "day")
    ]
    
    for multiplier, timeframe in timeframes:
        test_timeframe(fetcher, timeframe, multiplier)
        
if __name__ == "__main__":
    main()
