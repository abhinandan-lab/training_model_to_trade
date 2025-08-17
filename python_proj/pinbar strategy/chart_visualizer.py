import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime

class PinBarChartVisualizer:
    def __init__(self):
        self.style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(
                up='g', down='r',
                edge='inherit',
                wick={'up':'blue', 'down':'orange'},
                volume='in'
            ),
            gridstyle='-', 
            y_on_right=True
        )
    
    def prepare_chart_data(self, df):
        """Prepare dataframe for mplfinance visualization"""
        chart_df = df.copy()
        chart_df.set_index('time', inplace=True)
        chart_df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        return chart_df
    
    def create_pattern_addplot(self, df):
        """Create addplot objects for BOTH pin bars AND engulfing patterns"""
        addplots = []
        
        # PIN BAR MARKERS (Triangles)
        # Bullish pin bars (lime triangles pointing up)
        bullish_pins = df['bullish_pin_bar'].copy()
        bullish_pins[bullish_pins == 0] = np.nan
        bullish_pins[bullish_pins == 1] = df['Low'] * 0.9995  # Slightly below low
        
        if bullish_pins.notna().any():
            addplots.append(mpf.make_addplot(bullish_pins, 
                                           type='scatter', 
                                           markersize=100, 
                                           marker='^', 
                                           color='lime'))
        
        # Bearish pin bars (red triangles pointing down)  
        bearish_pins = df['bearish_pin_bar'].copy()
        bearish_pins[bearish_pins == 0] = np.nan
        bearish_pins[bearish_pins == 1] = df['High'] * 1.0005  # Slightly above high
        
        if bearish_pins.notna().any():
            addplots.append(mpf.make_addplot(bearish_pins, 
                                           type='scatter', 
                                           markersize=100, 
                                           marker='v', 
                                           color='red'))
        
        # ENGULFING PATTERN MARKERS (Squares)
        # Bullish engulfing (green squares)
        bullish_engulfing = df['bullish_engulfing'].copy()
        bullish_engulfing[bullish_engulfing == 0] = np.nan
        bullish_engulfing[bullish_engulfing == 1] = df['Low'] * 0.999  # Below pin bars
        
        if bullish_engulfing.notna().any():
            addplots.append(mpf.make_addplot(bullish_engulfing, 
                                           type='scatter', 
                                           markersize=80, 
                                           marker='s',  # Square marker
                                           color='darkgreen'))
        
        # Bearish engulfing (dark red squares)
        bearish_engulfing = df['bearish_engulfing'].copy()
        bearish_engulfing[bearish_engulfing == 0] = np.nan
        bearish_engulfing[bearish_engulfing == 1] = df['High'] * 1.001  # Above pin bars
        
        if bearish_engulfing.notna().any():
            addplots.append(mpf.make_addplot(bearish_engulfing, 
                                           type='scatter', 
                                           markersize=80, 
                                           marker='s',  # Square marker
                                           color='darkred'))
        
        return addplots if addplots else None
    
    def plot_training_data(self, processed_data, candles_to_show=100, save_path=None):
        """Plot training data with pin bars AND engulfing patterns highlighted"""
        print(f"📊 Plotting Training Data Chart with All Patterns...")
        
        # Prepare data
        chart_data = self.prepare_chart_data(processed_data.tail(candles_to_show))
        pattern_addplots = self.create_pattern_addplot(chart_data)
        
        # Count all patterns in this view
        bullish_pins = chart_data['bullish_pin_bar'].sum()
        bearish_pins = chart_data['bearish_pin_bar'].sum()
        bullish_eng = chart_data['bullish_engulfing'].sum()
        bearish_eng = chart_data['bearish_engulfing'].sum()
        
        total_patterns = bullish_pins + bearish_pins + bullish_eng + bearish_eng
        
        title = (f"EURUSD 15M - Training Data (Last {candles_to_show} candles)\n"
                f"Pin Bars: {bullish_pins + bearish_pins} (▲{bullish_pins} ▼{bearish_pins}) | "
                f"Engulfing: {bullish_eng + bearish_eng} (■{bullish_eng} ■{bearish_eng})")
        
        if save_path:
            mpf.plot(chart_data[['Open', 'High', 'Low', 'Close', 'Volume']], 
                    type='candle',
                    style=self.style,
                    title=title,
                    volume=True,
                    addplot=pattern_addplots,
                    savefig=save_path,
                    figsize=(16, 10))
            print(f"✅ Training chart saved to: {save_path}")
        else:
            mpf.plot(chart_data[['Open', 'High', 'Low', 'Close', 'Volume']], 
                    type='candle',
                    style=self.style,
                    title=title,
                    volume=True,
                    addplot=pattern_addplots,
                    figsize=(16, 10))
            print(f"✅ Training chart displayed with {total_patterns} patterns")
    
    def plot_testing_data(self, test_data_processed, candles_to_show=100, save_path=None):
        """Plot testing data with pin bars AND engulfing patterns highlighted"""
        print(f"📊 Plotting Testing Data Chart with All Patterns...")
        
        chart_data = self.prepare_chart_data(test_data_processed.tail(candles_to_show))
        pattern_addplots = self.create_pattern_addplot(chart_data)
        
        # Count all patterns
        bullish_pins = chart_data['bullish_pin_bar'].sum()
        bearish_pins = chart_data['bearish_pin_bar'].sum()
        bullish_eng = chart_data['bullish_engulfing'].sum()
        bearish_eng = chart_data['bearish_engulfing'].sum()
        
        total_patterns = bullish_pins + bearish_pins + bullish_eng + bearish_eng
        
        title = (f"EURUSD 15M - Testing Data (Last {candles_to_show} candles)\n"
                f"Pin Bars: {bullish_pins + bearish_pins} (▲{bullish_pins} ▼{bearish_pins}) | "
                f"Engulfing: {bullish_eng + bearish_eng} (■{bullish_eng} ■{bearish_eng})")
        
        if save_path:
            mpf.plot(chart_data[['Open', 'High', 'Low', 'Close', 'Volume']], 
                    type='candle',
                    style=self.style,
                    title=title,
                    volume=True,
                    addplot=pattern_addplots,
                    savefig=save_path,
                    figsize=(16, 10))
            print(f"✅ Testing chart saved to: {save_path}")
        else:
            mpf.plot(chart_data[['Open', 'High', 'Low', 'Close', 'Volume']], 
                    type='candle',
                    style=self.style,
                    title=title,
                    volume=True,
                    addplot=pattern_addplots,
                    figsize=(16, 10))
            print(f"✅ Testing chart displayed with {total_patterns} patterns")

# UPDATED PLUG-N-PLAY FUNCTIONS
def quick_training_chart(trainer, candles=100):
    """Quick function to chart training data with ALL patterns"""
    if trainer.processed_data is None:
        print("❌ No processed data available")
        return
    
    # Check if engulfing patterns exist in the data
    if 'bullish_engulfing' not in trainer.processed_data.columns:
        print("⚠️ Engulfing patterns not detected yet. Run process_training_data() with engulfing detection.")
        return
    
    try:
        visualizer = PinBarChartVisualizer()
        visualizer.plot_training_data(trainer.processed_data, candles)
    except Exception as e:
        print(f"⚠️ Chart error: {e}")

def quick_testing_chart(trainer, candles=100):
    """Quick function to chart testing data with ALL patterns"""
    if trainer.test_data is None:
        print("❌ No test data available")
        return
    
    try:
        # Process test data with pin bars AND engulfing patterns
        test_processed = trainer.detect_pin_bars(trainer.test_data)
        test_processed = trainer._detect_engulfing_patterns(test_processed)
        
        visualizer = PinBarChartVisualizer()
        visualizer.plot_testing_data(test_processed, candles)
    except Exception as e:
        print(f"⚠️ Chart error: {e}")

def show_pattern_legend():
    """Display legend for chart markers"""
    print("📊 Chart Pattern Legend:")
    print("▲ Lime Triangle: Bullish Pin Bar")
    print("▼ Red Triangle: Bearish Pin Bar") 
    print("■ Dark Green Square: Bullish Engulfing")
    print("■ Dark Red Square: Bearish Engulfing")
