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
plot.setLabel('left', 'Price')
plot.setLabel('bottom', 'Time')
win.show()

# Fetch latest candles
def fetch_candles():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Detect swing highs/lows
def detect_swings(df, lookback=3):
    highs = df['high']
    lows = df['low']
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows

# Draw chart + trendlines
def draw_chart():
    df = fetch_candles()
    plot.clear()

    for i in range(len(df)):
        t = i
        o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
        color = pg.mkColor('g') if c >= o else pg.mkColor('r')

        # Candle body
        rect = QGraphicsRectItem(t - 0.3, min(o, c), 0.6, abs(o - c))
        rect.setPen(pg.mkPen(color))
        rect.setBrush(pg.mkBrush(color))
        plot.addItem(rect)

        # Wick line
        wick = pg.PlotDataItem([t, t], [l, h], pen=pg.mkPen(color))
        plot.addItem(wick)

    # Detect swings
    swing_highs, swing_lows = detect_swings(df)

    # Trendline: swing highs
    if len(swing_highs) >= 2:
        x_vals = [swing_highs[-2][0], swing_highs[-1][0]]
        y_vals = [swing_highs[-2][1], swing_highs[-1][1]]
        plot.addItem(pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen('orange', width=2)))

    # Trendline: swing lows
    if len(swing_lows) >= 2:
        x_vals = [swing_lows[-2][0], swing_lows[-1][0]]
        y_vals = [swing_lows[-2][1], swing_lows[-1][1]]
        plot.addItem(pg.PlotDataItem(x_vals, y_vals, pen=pg.mkPen('cyan', width=2)))

# Timer to refresh every 10 seconds
timer = QtCore.QTimer()
timer.timeout.connect(draw_chart)
timer.start(1000)

draw_chart()
app.exec_()

mt5.shutdown()
