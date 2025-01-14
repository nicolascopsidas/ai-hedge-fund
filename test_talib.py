import numpy as np
import talib

# Create sample data
close_prices = np.random.random(100)

# Try to calculate a simple moving average
sma = talib.SMA(close_prices, timeperiod=20)

print("TA-Lib test successful!")
print(f"Sample SMA values: {sma[:5]}")
