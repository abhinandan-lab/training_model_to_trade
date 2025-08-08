import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Connect to MT5
mt5.initialize()

# Parameters
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 100

# Fetch historical data
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
mt5.shutdown()

# Create DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Plot with Plotly
fig = go.Figure(data=[go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

fig.update_layout(title='BTCUSD 1-Minute Candlestick Chart',
                  xaxis_title='Time',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False)

fig.show()
