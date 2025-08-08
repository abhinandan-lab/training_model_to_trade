import MetaTrader5 as mt5
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    exit()

symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 100

# Setup PyQt application
app = QtWidgets.QApplication([])

# Create main window
win = pg.GraphicsLayoutWidget(title="BTCUSD Live Chart")
win.resize(1200, 700)
win.show()

# Create plot
plot = win.addPlot(title="BTCUSD Candlestick Chart")
plot.showGrid(x=True, y=True)
plot.setLabel('left', 'Price ($)')
plot.setLabel('bottom', 'Candle Index')

# Create info label
info_label = pg.LabelItem(justify='left')
win.addItem(info_label, row=0, col=0)
info_label.setText("Loading data...")

# Store previous view range
previous_view_range = None

def fetch_data():
    """Fetch OHLC data from MT5"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        print(f"Data fetch error: {e}")
        return None

def create_candlestick_data(df):
    """Convert OHLC data to candlestick format for pyqtgraph"""
    data = []
    for i in range(len(df)):
        open_val = df.iloc[i]['open']
        high_val = df.iloc[i]['high'] 
        low_val = df.iloc[i]['low']
        close_val = df.iloc[i]['close']
        
        # Create candlestick data: [x, open, close, min, max]
        data.append([i, open_val, close_val, low_val, high_val])
    
    return np.array(data)

def update_chart():
    """Update the chart with new data"""
    global previous_view_range
    
    df = fetch_data()
    
    if df is None or len(df) == 0:
        info_label.setText("Error: No data available from MT5")
        return
    
    # Store current view range if user has moved the chart
    current_view_range = plot.getViewBox().viewRange()
    if previous_view_range is None:
        # First time - let it auto range
        should_auto_range = True
    else:
        # Check if user has manually moved the chart
        # If the view hasn't changed much from previous, user probably didn't move it
        x_diff = abs(current_view_range[0][0] - previous_view_range[0][0])
        y_diff = abs(current_view_range[1][0] - previous_view_range[1][0])
        
        # If changes are minimal, probably just from auto-update
        should_auto_range = x_diff < 1.0 and y_diff < (df['high'].max() - df['low'].min()) * 0.1
    
    # Clear previous plot
    plot.clear()
    
    # Get candlestick data
    candlestick_data = create_candlestick_data(df)
    
    # Draw candlesticks manually
    for i, (x, open_val, close_val, low_val, high_val) in enumerate(candlestick_data):
        # Determine color
        color = 'g' if close_val >= open_val else 'r'
        
        # Draw wick (high-low line)
        plot.plot([x, x], [low_val, high_val], pen=pg.mkPen(color, width=1))
        
        # Draw body
        body_height = abs(close_val - open_val)
        if body_height > 0:
            # Create rectangle for body
            body_y = min(open_val, close_val)
            
            # Create rectangle item - both red and green candles are filled
            rect = pg.QtWidgets.QGraphicsRectItem(x-0.3, body_y, 0.6, body_height)
            rect.setPen(pg.mkPen(color, width=1))
            rect.setBrush(pg.mkBrush(color))  # Fill both red and green candles
            plot.addItem(rect)
        else:
            # Doji - draw horizontal line
            plot.plot([x-0.3, x+0.3], [open_val, open_val], pen=pg.mkPen(color, width=2))
    
    # Update info
    last_price = df['close'].iloc[-1]
    high_price = df['high'].max()
    low_price = df['low'].min() 
    first_price = df['open'].iloc[0]
    
    change = last_price - first_price
    change_pct = (change / first_price) * 100 if first_price != 0 else 0
    
    info_text = (
        f"<span style='color: white; font-size: 12pt; background-color: rgba(0,0,0,150);'>"
        f"<b>{symbol}</b><br>"
        f"Last: ${last_price:.2f}<br>"
        f"High: ${high_price:.2f}<br>"
        f"Low: ${low_price:.2f}<br>"
        f"Change: {change:+.2f} ({change_pct:+.2f}%)"
        f"</span>"
    )
    
    info_label.setText(info_text)
    
    # Handle view range
    if should_auto_range:
        plot.autoRange()
        # Store the new auto-ranged view
        previous_view_range = plot.getViewBox().viewRange()
    else:
        # Restore user's view range
        plot.getViewBox().setRange(xRange=current_view_range[0], yRange=current_view_range[1], padding=0)
        previous_view_range = current_view_range
    
    print(f"Chart updated - Last price: ${last_price:.2f}")

# Create timer for updates
timer = QtCore.QTimer()
timer.timeout.connect(update_chart)
timer.start(2000)  # Update every 2 seconds

# Initial update
update_chart()

# Run application
try:
    app.exec_()
except KeyboardInterrupt:
    print("Application interrupted")
finally:
    timer.stop()
    mt5.shutdown()
    print("MT5 connection closed")