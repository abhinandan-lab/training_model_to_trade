import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Initialize MT5
mt5.initialize()

# Define symbol and timeframe
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1  # 1-minute
n_candles = 100

# Get data
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)

# Shutdown MT5
mt5.shutdown()

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

print(df.tail())
