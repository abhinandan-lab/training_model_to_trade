import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Initialize
mt5.initialize()

# Config
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 1000

# Fetch
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
mt5.shutdown()

# Convert
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Save
csv_path = f"{symbol}_{timeframe}_candles.csv"
df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")
