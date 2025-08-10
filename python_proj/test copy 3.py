import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# ================= CONFIG =================
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M5
num_bars = 2_000
fractal_length = 5
bull_color = "green"
bear_color = "red"
sr_linestyle = "dashed"
sr_alpha = 0.6
marker_size = 120
show_sr_lines = True
show_plot = True
save_path = None  # Set to "chart.png" if you want to save
style = "yahoo"
title_suffix = "LuxAlgo-like Fractal BOS/CHoCH"
# ==========================================

# Initialize MT5
if not mt5.initialize():
    raise RuntimeError("MT5 initialize() failed")

# Get historical data
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
if rates is None or len(rates) == 0:
    mt5.shutdown()
    raise RuntimeError("No data returned from MT5")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# Keep only needed columns, enforce dtypes
df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
df = df.astype({
    "open": "float64",
    "high": "float64", 
    "low": "float64",
    "close": "float64",
    "tick_volume": "int64",
})

# Add columns with explicit categories
df["fract_type"] = pd.Categorical(values=[None] * len(df), categories=["bull", "bear"])
df["event"] = pd.Categorical(
    values=[None] * len(df),
    categories=["BOS_bull", "BOS_bear", "CHoCH_bull", "CHoCH_bear"],
)

# --- Detect fractals ---
def find_fractals(data: pd.DataFrame, length: int = 5):
    highs_idx = []
    lows_idx = []
    H = data["high"].to_numpy()
    L = data["low"].to_numpy()
    n = len(data)

    for i in range(length, n - length):
        window_h = H[i - length : i + length + 1]
        window_l = L[i - length : i + length + 1]

        if H[i] == np.max(window_h):
            highs_idx.append(i)
            data.loc[data.index[i], "fract_type"] = "bear"
        elif L[i] == np.min(window_l):
            lows_idx.append(i)
            data.loc[data.index[i], "fract_type"] = "bull"

    return highs_idx, lows_idx

highs_idx, lows_idx = find_fractals(df, fractal_length)

# --- Market structure events ---
events = []
last_dir = None

for i in range(len(df)):
    f = df.loc[df.index[i], "fract_type"]
    if pd.isna(f):
        continue

    if f == "bear":
        direction = "bear"
        price = float(df.loc[df.index[i], "low"])
        if last_dir == "bull":
            events.append({"bar_idx": i, "type": "CHoCH", "direction": direction, "price": price})
            df.loc[df.index[i], "event"] = "CHoCH_bear"
        else:
            events.append({"bar_idx": i, "type": "BOS", "direction": direction, "price": price})
            df.loc[df.index[i], "event"] = "BOS_bear"
        last_dir = direction

    elif f == "bull":
        direction = "bull"
        price = float(df.loc[df.index[i], "high"])
        if last_dir == "bear":
            events.append({"bar_idx": i, "type": "CHoCH", "direction": direction, "price": price})
            df.loc[df.index[i], "event"] = "CHoCH_bull"
        else:
            events.append({"bar_idx": i, "type": "BOS", "direction": direction, "price": price})
            df.loc[df.index[i], "event"] = "BOS_bull"
        last_dir = direction

# --- Plotting ---
if show_plot:
    apds = []

    # Raw fractal markers
    bull_fract_y = [df["low"].iloc[i] if df.loc[df.index[i], "fract_type"] == "bull" else None for i in range(len(df))]
    bear_fract_y = [df["high"].iloc[i] if df.loc[df.index[i], "fract_type"] == "bear" else None for i in range(len(df))]

    apds.append(mpf.make_addplot(bull_fract_y, type="scatter", markersize=40, marker="^", color=bull_color))
    apds.append(mpf.make_addplot(bear_fract_y, type="scatter", markersize=40, marker="v", color=bear_color))

    # BOS/CHoCH markers and hlines
    prices = []
    colors = []
    for e in events:
        idx = e["bar_idx"]
        price = e["price"]
        direction = e["direction"]
        color = bull_color if direction == "bull" else bear_color

        marker_series = [price if i == idx else None for i in range(len(df))]
        apds.append(mpf.make_addplot(
            marker_series,
            type="scatter",
            markersize=marker_size,
            marker="^" if direction == "bull" else "v",
            color=color,
        ))

        if show_sr_lines:
            prices.append(price)
            colors.append(color)

    hline_kwargs = {}
    if show_sr_lines and prices:
        hline_kwargs["hlines"] = dict(
            hlines=prices,
            colors=colors,
            linewidths=1,
            linestyle=sr_linestyle,
            alpha=sr_alpha,
        )

    df_plot = df.set_index("time")

    # Build plot kwargs conditionally - FIXED VERSION
    plot_kwargs = {
        "type": "candle",
        "style": style,
        "addplot": apds,
        "title": f"{symbol} - Fractal Market Structure (BOS/CHoCH) - {title_suffix}",
        "ylabel": "Price",
        "volume": False,
        **hline_kwargs,
    }
    
    # Only add savefig if save_path is not None
    if save_path:
        plot_kwargs["savefig"] = save_path

    mpf.plot(df_plot, **plot_kwargs)

    if not save_path:
        plt.show()

# Console output
print(f"Symbol: {symbol}")
print(f"Timeframe: {timeframe}, Bars: {len(df)}")
print(f"Fractal length: {fractal_length}")
print(f"Total fractals found: {len(highs_idx) + len(lows_idx)} (bullish lows: {len(lows_idx)}, bearish highs: {len(highs_idx)})")
print(f"Total events (BOS/CHoCH): {len(events)}")

if events:
    events_df = pd.DataFrame(events)
    print("\nSample events:")
    print(events_df.head())

mt5.shutdown()
