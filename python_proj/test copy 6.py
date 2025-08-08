import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

# Connect
mt5.initialize()

symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 100

def get_candles():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

try:
    for _ in range(10):  # Loop 10 times (approx. 2.5 minutes)
        df = get_candles()

        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])

        fig.update_layout(title='BTCUSD Live Chart (auto-refresh)',
                          xaxis_title='Time',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        fig.show()
        print("Chart updated. Waiting 15 seconds...")
        time.sleep(15)

finally:
    mt5.shutdown()
