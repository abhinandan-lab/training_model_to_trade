import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time

# Connect to MT5
mt5.initialize()

# Config
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
max_candles = 100

# Load initial history
df = pd.DataFrame(mt5.copy_rates_from_pos(symbol, timeframe, 0, max_candles))
df['time'] = pd.to_datetime(df['time'], unit='s')

# Stream for 30 seconds
print("Streaming live candles...")
start = time.time()
while time.time() - start < 30:
    latest = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    if latest:
        latest_df = pd.DataFrame(latest)
        latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
        if latest_df['time'].iloc[0] != df['time'].iloc[-1]:
            df = pd.concat([df, latest_df]).tail(max_candles).reset_index(drop=True)
            print(df.tail(1)[['time', 'open', 'high', 'low', 'close']])
    time.sleep(1)

# Shutdown
mt5.shutdown()
