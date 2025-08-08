import MetaTrader5 as mt5
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QGraphicsRectItem
from datetime import datetime

# Connect to MT5
mt5.initialize()
symbol = "BTCUSD"
timeframe = mt5.TIMEFRAME_M1
n_candles = 100

# PyQtGraph window
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="BTCUSD Live Candlestick Chart")
win.resize(1000, 600)
plot = win.addPlot()
plot.showGrid(x=True, y=True)
plot.setLabel('left', 'Price')
plot.setLabel('bottom', 'Time')
win.show()

def fetch_candles():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def draw_chart():
    df = fetch_candles()
    plot.clear()

    for i in range(len(df)):
        t = i
        o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
        color = pg.mkColor('g') if c >= o else pg.mkColor('r')

        # Candle body (rect)
        rect = QGraphicsRectItem(t - 0.3, min(o, c), 0.6, abs(o - c))
        rect.setPen(pg.mkPen(color))
        rect.setBrush(pg.mkBrush(color))
        plot.addItem(rect)

        # Wick (line)
        wick = pg.PlotDataItem([t, t], [l, h], pen=pg.mkPen(color))
        plot.addItem(wick)

# Update every 10 seconds
timer = QtCore.QTimer()
timer.timeout.connect(draw_chart)
# timer.start(10000)
timer.start(1000)

draw_chart()
app.exec_()

mt5.shutdown()
