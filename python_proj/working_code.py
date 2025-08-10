import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt

# ================= CONFIG =================
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M5
num_bars = 1_000
fractal_length = 5
bull_color = "#089981"  # LuxAlgo green
bear_color = "#f23645"  # LuxAlgo red
line_alpha = 0.8
marker_size = 60
show_fractal_markers = True
show_plot = True
save_path = None
style = "yahoo"
title_suffix = "LuxAlgo BOS/CHoCH Fixed Logic"
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
df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()
df = df.astype({
    "open": "float64", "high": "float64", "low": "float64", 
    "close": "float64", "tick_volume": "int64",
})

# --- Proper Fractal Detection (matching Pine Script) ---
def detect_fractals_pine_style(data: pd.DataFrame, length: int = 5):
    """
    Detects fractals using Pine Script logic:
    - For swing high: high[p] == ta.highest(length) where p = length/2
    - For swing low: low[p] == ta.lowest(length) where p = length/2
    """
    p = length // 2
    fractals = []
    
    for i in range(p, len(data) - p):
        # Check for swing high fractal
        window_high = data['high'].iloc[i-p:i+p+1]
        if data['high'].iloc[i] == window_high.max():
            fractals.append({
                "idx": i,
                "type": "high",
                "price": data['high'].iloc[i],
                "is_crossed": False
            })
        
        # Check for swing low fractal  
        window_low = data['low'].iloc[i-p:i+p+1]
        if data['low'].iloc[i] == window_low.min():
            fractals.append({
                "idx": i,
                "type": "low", 
                "price": data['low'].iloc[i],
                "is_crossed": False
            })
    
    return fractals

fractals = detect_fractals_pine_style(df, fractal_length)

# --- BOS/CHoCH Detection with Pine Script Logic ---
events = []
lines_data = []
labels_data = []

# Market structure state (matches Pine Script 'os' variable)
market_structure = 0  # 0 = neutral, 1 = bullish, -1 = bearish

# Active fractals waiting to be broken
active_fractals = [f.copy() for f in fractals]

for i in range(len(df)):
    current_close = df["close"].iloc[i]
    
    # Check for bullish breakouts (close crosses above swing high)
    for fractal in active_fractals:
        if (fractal["type"] == "high" and 
            not fractal["is_crossed"] and 
            fractal["idx"] < i and
            current_close > fractal["price"]):
            
            # Determine BOS vs CHoCH based on previous market structure
            if market_structure == -1:  # Previous was bearish
                event_type = "CHoCH"
            else:  # Previous was bullish or neutral
                event_type = "BOS"
            
            # Create event
            events.append({
                "fractal_idx": fractal["idx"],
                "break_idx": i,
                "fractal_price": fractal["price"],
                "type": event_type,
                "direction": "bullish"
            })
            
            # Create line from fractal to breakout
            lines_data.append({
                "start_idx": fractal["idx"],
                "end_idx": i,
                "price": fractal["price"],
                "color": bull_color,
                "type": event_type
            })
            
            # Create label at midpoint
            mid_idx = (fractal["idx"] + i) // 2
            labels_data.append({
                "idx": mid_idx,
                "price": fractal["price"],
                "text": event_type,
                "color": bull_color,
                "direction": "bullish"
            })
            
            # Update market structure and mark fractal as crossed
            market_structure = 1  # Now bullish
            fractal["is_crossed"] = True
            
            break  # Only one breakout per bar
    
    # Check for bearish breakouts (close crosses below swing low)  
    for fractal in active_fractals:
        if (fractal["type"] == "low" and 
            not fractal["is_crossed"] and 
            fractal["idx"] < i and
            current_close < fractal["price"]):
            
            # Determine BOS vs CHoCH based on previous market structure
            if market_structure == 1:  # Previous was bullish
                event_type = "CHoCH"
            else:  # Previous was bearish or neutral
                event_type = "BOS"
            
            # Create event
            events.append({
                "fractal_idx": fractal["idx"],
                "break_idx": i,
                "fractal_price": fractal["price"],
                "type": event_type,
                "direction": "bearish"
            })
            
            # Create line from fractal to breakout
            lines_data.append({
                "start_idx": fractal["idx"],
                "end_idx": i,
                "price": fractal["price"],
                "color": bear_color,
                "type": event_type
            })
            
            # Create label at midpoint
            mid_idx = (fractal["idx"] + i) // 2
            labels_data.append({
                "idx": mid_idx,
                "price": fractal["price"],
                "text": event_type,
                "color": bear_color,
                "direction": "bearish"
            })
            
            # Update market structure and mark fractal as crossed
            market_structure = -1  # Now bearish
            fractal["is_crossed"] = True
            
            break  # Only one breakout per bar

# --- Plotting ---
if show_plot:
    apds = []

    # Fractal markers
    if show_fractal_markers:
        for fractal in fractals:
            marker_series = np.array([fractal["price"] if i == fractal["idx"] else np.nan for i in range(len(df))], dtype=float)
            if fractal["type"] == "high":
                apds.append(mpf.make_addplot(marker_series, type="scatter", markersize=25, marker="v", color=bear_color, alpha=0.4))
            else:  # low
                apds.append(mpf.make_addplot(marker_series, type="scatter", markersize=25, marker="^", color=bull_color, alpha=0.4))

    # BOS/CHoCH lines (from fractal to breakout point)
    for line_data in lines_data:
        start_idx = line_data["start_idx"]
        end_idx = line_data["end_idx"] 
        price = line_data["price"]
        color = line_data["color"]
        
        # Line only between fractal and breakout
        line_series = np.full(len(df), np.nan, dtype=float)
        line_series[start_idx:end_idx+1] = price
        
        apds.append(mpf.make_addplot(
            line_series,
            type="line",
            color=color,
            alpha=line_alpha,
            width=2
        ))

    # BOS/CHoCH markers at breakout points
    for event in events:
        break_idx = event["break_idx"]
        price = event["fractal_price"]
        direction = event["direction"]
        color = bull_color if direction == "bullish" else bear_color
        
        marker_series = np.array([price if i == break_idx else np.nan for i in range(len(df))], dtype=float)
        apds.append(mpf.make_addplot(
            marker_series,
            type="scatter",
            markersize=marker_size,
            marker="^" if direction == "bullish" else "v",
            color=color,
            alpha=0.9
        ))

    # Labels (using shapes: square=CHoCH, circle=BOS)
    for label_data in labels_data:
        idx = label_data["idx"]
        price = label_data["price"]
        text = label_data["text"]
        color = label_data["color"]
        
        marker = "s" if text == "CHoCH" else "o"
        
        label_series = np.array([price if i == idx else np.nan for i in range(len(df))], dtype=float)
        apds.append(mpf.make_addplot(
            label_series,
            type="scatter",
            markersize=40,
            marker=marker,
            color="white",
            edgecolors=color,
            alpha=0.9
        ))

    df_plot = df.set_index("time")

    plot_kwargs = {
        "type": "candle",
        "style": style,
        "addplot": apds,
        "title": f"{symbol} - Fixed BOS/CHoCH Detection Logic",
        "ylabel": "Price",
        "volume": False,
        "warn_too_much_data": num_bars + 100,
    }
    
    if save_path:
        plot_kwargs["savefig"] = save_path

    mpf.plot(df_plot, **plot_kwargs)

    if not save_path:
        plt.show()

# Detailed output
print(f"Symbol: {symbol}")
print(f"Timeframe: {timeframe}, Bars: {len(df)}")
print(f"Fractal length: {fractal_length}")
print(f"Total fractals found: {len(fractals)}")
print(f"Total BOS/CHoCH events: {len(events)}")

if events:
    print("\n=== BOS/CHoCH Events ===")
    for i, event in enumerate(events):
        fractal_bar = event['fractal_idx']
        break_bar = event['break_idx']
        bars_to_break = break_bar - fractal_bar
        print(f"{i+1:2d}. {event['type']:5s} ({event['direction']:7s}) | "
              f"Fractal: {fractal_bar:4d} | Break: {break_bar:4d} | "
              f"Bars to break: {bars_to_break:3d} | Price: {event['fractal_price']:8.2f}")

print(f"\nBOS events: {len([e for e in events if e['type'] == 'BOS'])}")
print(f"CHoCH events: {len([e for e in events if e['type'] == 'CHoCH'])}")

print("\n=== Legend ===")
print("◦ Small triangles: Raw fractals (^=low, v=high)")
print("◦ Large triangles: BOS/CHoCH breakout points")
print("◦ White circles with colored border: BOS events")
print("◦ White squares with colored border: CHoCH events")
print("◦ Solid lines: Connect fractal to breakout point")

mt5.shutdown()
