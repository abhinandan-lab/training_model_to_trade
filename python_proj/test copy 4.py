import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# ================= CONFIG =================
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M5
num_bars = 1_000  # Reduced for better visibility
fractal_length = 5
bull_color = "green"
bear_color = "red"
sr_linestyle = "solid"  # Solid lines like LuxAlgo
sr_alpha = 0.8
marker_size = 80
show_bos_choch_lines = True  # Show connecting lines
show_plot = True
save_path = None
style = "yahoo"
title_suffix = "LuxAlgo-style BOS/CHoCH Lines"
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

# --- Market structure with fractal tracking ---
events = []
lines_data = []  # Store line data for BOS/CHoCH connections
last_dir = None
last_bull_fractal = None  # Track last bullish fractal (swing low)
last_bear_fractal = None  # Track last bearish fractal (swing high)

for i in range(len(df)):
    f = df.loc[df.index[i], "fract_type"]
    
    # Update fractal tracking
    if f == "bull":  # Swing low fractal
        last_bull_fractal = {"idx": i, "price": df.loc[df.index[i], "low"]}
    elif f == "bear":  # Swing high fractal  
        last_bear_fractal = {"idx": i, "price": df.loc[df.index[i], "high"]}
    
    # Check for structure breaks
    if pd.notna(f):
        if f == "bear":  # Swing high -> potential bearish break
            direction = "bear"
            break_price = float(df.loc[df.index[i], "low"])
            
            if last_dir == "bull":
                event_type = "CHoCH"
                df.loc[df.index[i], "event"] = "CHoCH_bear"
            else:
                event_type = "BOS"
                df.loc[df.index[i], "event"] = "BOS_bear"
                
            events.append({
                "bar_idx": i, 
                "type": event_type, 
                "direction": direction, 
                "price": break_price,
                "fractal_idx": i,  # The fractal that caused the break
                "fractal_price": df.loc[df.index[i], "high"]  # High of the swing high fractal
            })
            
            # Store line data: from fractal point to break point
            if show_bos_choch_lines:
                lines_data.append({
                    "start_idx": i,
                    "start_price": df.loc[df.index[i], "high"],
                    "end_idx": i,  # For now, same as start - will extend as price moves
                    "end_price": df.loc[df.index[i], "high"],
                    "color": bear_color,
                    "type": event_type,
                    "direction": direction
                })
            
            last_dir = direction

        elif f == "bull":  # Swing low -> potential bullish break
            direction = "bull"
            break_price = float(df.loc[df.index[i], "high"])
            
            if last_dir == "bear":
                event_type = "CHoCH"
                df.loc[df.index[i], "event"] = "CHoCH_bull"
            else:
                event_type = "BOS"
                df.loc[df.index[i], "event"] = "BOS_bull"
                
            events.append({
                "bar_idx": i, 
                "type": event_type, 
                "direction": direction, 
                "price": break_price,
                "fractal_idx": i,
                "fractal_price": df.loc[df.index[i], "low"]
            })
            
            # Store line data: from fractal point to break point
            if show_bos_choch_lines:
                lines_data.append({
                    "start_idx": i,
                    "start_price": df.loc[df.index[i], "low"],
                    "end_idx": i,
                    "end_price": df.loc[df.index[i], "low"], 
                    "color": bull_color,
                    "type": event_type,
                    "direction": direction
                })
            
            last_dir = direction

# --- Plotting ---
if show_plot:
    apds = []

    # Raw fractal markers (small)
    bull_fract_y = np.array([df["low"].iloc[i] if df.loc[df.index[i], "fract_type"] == "bull" else np.nan for i in range(len(df))], dtype=float)
    bear_fract_y = np.array([df["high"].iloc[i] if df.loc[df.index[i], "fract_type"] == "bear" else np.nan for i in range(len(df))], dtype=float)

    apds.append(mpf.make_addplot(bull_fract_y, type="scatter", markersize=30, marker="^", color=bull_color, alpha=0.6))
    apds.append(mpf.make_addplot(bear_fract_y, type="scatter", markersize=30, marker="v", color=bear_color, alpha=0.6))

    # BOS/CHoCH markers (larger)
    for e in events:
        idx = e["bar_idx"]
        price = e["fractal_price"]  # Use fractal price for marker position
        direction = e["direction"]
        color = bull_color if direction == "bull" else bear_color

        marker_series = np.array([price if i == idx else np.nan for i in range(len(df))], dtype=float)
        apds.append(mpf.make_addplot(
            marker_series,
            type="scatter",
            markersize=marker_size,
            marker="^" if direction == "bull" else "v",
            color=color,
        ))

    # BOS/CHoCH horizontal lines (like Pine Script)
    if show_bos_choch_lines and lines_data:
        for line_data in lines_data:
            start_idx = line_data["start_idx"]
            price = line_data["start_price"]
            color = line_data["color"]
            
            # Create horizontal line from fractal to current bar (extending to right)
            # In Pine Script: line.new(fractal_loc, fractal_value, current_bar, fractal_value)
            line_series = np.full(len(df), np.nan, dtype=float)
            # Fill from start_idx to end of data with the price level
            line_series[start_idx:] = price
            
            apds.append(mpf.make_addplot(
                line_series,
                type="line",
                color=color,
                linestyle=sr_linestyle,
                alpha=sr_alpha,
                width=1
            ))

    df_plot = df.set_index("time")

    # Build plot kwargs
    plot_kwargs = {
        "type": "candle",
        "style": style,
        "addplot": apds,
        "title": f"{symbol} - Market Structure BOS/CHoCH Lines - {title_suffix}",
        "ylabel": "Price",
        "volume": False,
        "warn_too_much_data": num_bars + 100,
    }
    
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
print(f"Total lines drawn: {len(lines_data)}")

if events:
    events_df = pd.DataFrame(events)
    print("\nSample events:")
    print(events_df.head())

mt5.shutdown()
