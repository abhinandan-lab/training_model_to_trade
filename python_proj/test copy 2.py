import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime, timedelta
import mplfinance as mpf
import numpy as np

# --- SETTINGS ---
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
NUM_CANDLES = 200

# --- INITIALIZE MT5 ---
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    mt5.shutdown()
    exit()

print("✅ MT5 initialized")

# --- FETCH DATA ---
utc_from = datetime.now(pytz.utc) - timedelta(minutes=NUM_CANDLES * 5)
utc_to = datetime.now(pytz.utc)
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, utc_from, utc_to)

if rates is None or len(rates) == 0:
    print("❌ No data fetched.")
    mt5.shutdown()
    exit()

# --- DATAFRAME ---
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.rename(columns={'tick_volume': 'volume'}, inplace=True)
df.set_index('time', inplace=True)

# --- SWING DETECTION ---
df['swing_high'] = np.nan
df['swing_low'] = np.nan

for i in range(1, len(df) - 1):
    if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
        df['swing_high'].iloc[i] = df['high'].iloc[i]
    if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
        df['swing_low'].iloc[i] = df['low'].iloc[i]

print(df[['open', 'high', 'low', 'close', 'swing_high', 'swing_low']].head(15))

# --- PLOT WITH MARKERS ---
mt5_style = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=mpf.make_marketcolors(
        up='lime',
        down='red',
        edge='inherit',
        wick='white',
        volume='in'
    ),
    facecolor='black',
    gridcolor='dimgray',
    gridstyle='-',
    rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'}
)

# Create swing point markers
ap = []
if df['swing_high'].notna().any():
    ap.append(mpf.make_addplot(df['swing_high'], type='scatter', markersize=100, marker='^', color='yellow'))
if df['swing_low'].notna().any():
    ap.append(mpf.make_addplot(df['swing_low'], type='scatter', markersize=100, marker='v', color='cyan'))

mpf.plot(df, type='candle', style=mt5_style, title=f"{SYMBOL} - Swing Points", volume=True, addplot=ap)

# --- SHUTDOWN ---
mt5.shutdown()
