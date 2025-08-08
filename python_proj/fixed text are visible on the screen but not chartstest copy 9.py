import MetaTrader5 as mt5
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsRectItem
from datetime import datetime

# Initialize MT5
mt5.initialize()
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 100

# Setup PyQt window
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="BTCUSD Live Candlestick Chart")
win.resize(1000, 600)
plot = win.addPlot()
plot.showGrid(x=True, y=True)
win.show()

# Create a text item for custom info with better styling
custom_text = pg.TextItem(
    text="Loading...", 
    color=(255, 255, 255),  # White color
    fill=(0, 0, 0, 150),    # Semi-transparent black background
    anchor=(0, 0)           # Top-left anchor
)

# Add text to the plot
plot.addItem(custom_text)

# Function to keep text at fixed screen position
def update_text_position():
    # Get the view box and its range
    view_range = plot.viewRange()
    
    # Calculate position in data coordinates (top-left with padding)
    x_range = view_range[0][1] - view_range[0][0]
    y_range = view_range[1][1] - view_range[1][0]
    
    x_pos = view_range[0][0] + x_range * 0.02  # 2% from left
    y_pos = view_range[1][1] - y_range * 0.02  # 2% from top
    
    custom_text.setPos(x_pos, y_pos)

# Connect view change signal to update text position
plot.sigRangeChanged.connect(update_text_position)

# Fetch candles from MT5
def fetch_candles():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Draw the chart and update custom text
def draw_chart():
    df = fetch_candles()
    
    # Clear only the candlestick items, preserve the text
    plot.clearPlots()
    
    # Update text content
    last_price = df['close'].iloc[-1]
    high_price = df['high'].max()
    low_price = df['low'].min()
    
    # Add more useful information
    price_change = df['close'].iloc[-1] - df['open'].iloc[0]
    price_change_pct = (price_change / df['open'].iloc[0]) * 100
    
    custom_text.setText(f"Symbol: {symbol}\n"
                       f"Last: ${last_price:.2f}\n"
                       f"High: ${high_price:.2f}\n"
                       f"Low: ${low_price:.2f}\n"
                       f"Change: {price_change:+.2f} ({price_change_pct:+.2f}%)")

    # Draw candles
    candle_items = []
    for i in range(len(df)):
        t = i
        o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
        color = pg.mkColor('g') if c >= o else pg.mkColor('r')

        # Candle body
        if abs(o - c) > 0:  # Only draw body if there's a difference
            rect = QGraphicsRectItem(t - 0.3, min(o, c), 0.6, abs(o - c))
            rect.setPen(pg.mkPen(color))
            rect.setBrush(pg.mkBrush(color))
            plot.addItem(rect)
            candle_items.append(rect)

        # Wick line
        wick = pg.PlotDataItem([t, t], [l, h], pen=pg.mkPen(color, width=1))
        plot.addItem(wick)
        candle_items.append(wick)
    
    # Update text position after drawing
    update_text_position()
    
    # Ensure text is on top
    custom_text.setZValue(1000)

# Timer to update chart
timer = QtCore.QTimer()
timer.timeout.connect(draw_chart)
timer.start(1000)  # Update every 1 second

# Initial draw and start app
draw_chart()
app.exec_()

# Shutdown MT5
mt5.shutdown()