import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

class PinBarTrainer:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.processed_data = None
    
    # STEP 1: MT5 Setup and Data Collection
    def setup_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        print("✅ MT5 connected successfully")
        return True
    
    def collect_data(self, total_candles=2000):
        """Collect EURUSD 15M data"""
        print("📡 Collecting EURUSD 15M data...")
        
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, total_candles)
        
        if rates is None:
            print("❌ No data received")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
        
        # Split train/test
        self.train_data = df.iloc[:1000].copy()
        self.test_data = df.iloc[1000:].copy()
        
        print(f"✅ Collected {len(df)} candles")
        print(f"📊 Training: {len(self.train_data)} | Testing: {len(self.test_data)}")
        return True
    
    # STEP 2: Pin Bar Detection
    def detect_pin_bars(self, df, min_nose_ratio=0.6, max_body_ratio=0.3):
        """Add pin bar detection to dataframe - FIXED VERSION"""
        df = df.copy()
        
        # Calculate candle components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['total_range'] = df['total_range'].replace(0, 0.0001)
        
        # Calculate ratios
        df['body_ratio'] = df['body_size'] / df['total_range']
        df['upper_nose_ratio'] = df['upper_shadow'] / df['total_range']
        df['lower_nose_ratio'] = df['lower_shadow'] / df['total_range']
        
        # Pin bar detection
        bullish_conditions = (
            (df['lower_nose_ratio'] >= min_nose_ratio) &
            (df['body_ratio'] <= max_body_ratio) &
            (df['upper_shadow'] <= df['body_size'])
        )
        
        bearish_conditions = (
            (df['upper_nose_ratio'] >= min_nose_ratio) &
            (df['body_ratio'] <= max_body_ratio) &
            (df['lower_shadow'] <= df['body_size'])
        )
        
        # Create columns
        df['bullish_pin_bar'] = bullish_conditions.astype(int)
        df['bearish_pin_bar'] = bearish_conditions.astype(int)
        df['any_pin_bar'] = (bullish_conditions | bearish_conditions).astype(int)
        
        # FIXED: Strength scoring 
        df['pin_bar_strength'] = 0
        
        # Calculate rolling mean once for the entire dataframe
        df['range_rolling_mean'] = df['total_range'].rolling(10, min_periods=1).mean()
        
        # Strength scoring for bullish pins - FIXED VERSION
        bullish_mask = df['bullish_pin_bar'] == 1
        if bullish_mask.any():
            df.loc[bullish_mask, 'pin_bar_strength'] = (
                (df.loc[bullish_mask, 'lower_nose_ratio'] >= 0.7).astype(int) +
                (df.loc[bullish_mask, 'body_ratio'] <= 0.2).astype(int) +
                (df.loc[bullish_mask, 'total_range'] > df.loc[bullish_mask, 'range_rolling_mean']).astype(int)
            )
        
        # Strength scoring for bearish pins - FIXED VERSION
        bearish_mask = df['bearish_pin_bar'] == 1
        if bearish_mask.any():
            df.loc[bearish_mask, 'pin_bar_strength'] = (
                (df.loc[bearish_mask, 'upper_nose_ratio'] >= 0.7).astype(int) +
                (df.loc[bearish_mask, 'body_ratio'] <= 0.2).astype(int) +
                (df.loc[bearish_mask, 'total_range'] > df.loc[bearish_mask, 'range_rolling_mean']).astype(int)
            )
        
        # Clean up temporary column
        df.drop('range_rolling_mean', axis=1, inplace=True)
        
        return df

    # STEP 3: Process All Data - ENHANCED VERSION
    def process_training_data(self):
        """Process training data with pin bar and engulfing pattern detection"""
        if self.train_data is None:
            print("❌ No training data available. Run collect_data() first.")
            return False
        
        print("🔧 Processing training data...")
        
        # Step 3a: Detect pin bars
        self.processed_data = self.detect_pin_bars(self.train_data)
        
        # Step 3b: Add engulfing pattern detection
        self.processed_data = self._detect_engulfing_patterns(self.processed_data)
        
        # Show pin bar results
        total_pins = self.processed_data['any_pin_bar'].sum()
        bullish_pins = self.processed_data['bullish_pin_bar'].sum()
        bearish_pins = self.processed_data['bearish_pin_bar'].sum()
        
        print(f"📊 Pin Bar Results:")
        print(f"   Total Pin Bars: {total_pins}")
        print(f"   Bullish: {bullish_pins} | Bearish: {bearish_pins}")
        print(f"   Frequency: {total_pins/len(self.processed_data)*100:.1f}%")
        
        # Show pin bar strength distribution
        if total_pins > 0:
            strength_counts = self.processed_data[self.processed_data['any_pin_bar']==1]['pin_bar_strength'].value_counts().sort_index()
            print(f"   Strength Distribution:")
            for strength, count in strength_counts.items():
                print(f"      Strength {strength}: {count} pin bars")
        
        # Show engulfing pattern results
        total_engulfing = self.processed_data['any_engulfing'].sum()
        bullish_engulfing = self.processed_data['bullish_engulfing'].sum()
        bearish_engulfing = self.processed_data['bearish_engulfing'].sum()
        
        print(f"\n📊 Engulfing Pattern Results:")
        print(f"   Total Engulfing: {total_engulfing}")
        print(f"   Bullish Engulfing: {bullish_engulfing}")
        print(f"   Bearish Engulfing: {bearish_engulfing}")
        print(f"   Frequency: {total_engulfing/len(self.processed_data)*100:.1f}%")
        
        # Show combination patterns
        combo_patterns = self.processed_data[
            (self.processed_data['any_pin_bar'] == 1) & 
            (self.processed_data['any_engulfing'] == 1)
        ]
        print(f"\n🎯 Combo Patterns (Pin Bar + Engulfing): {len(combo_patterns)}")
        
        # Show any high-quality patterns
        high_quality_pins = self.processed_data[
            (self.processed_data['any_pin_bar'] == 1) & 
            (self.processed_data['pin_bar_strength'] >= 2)
        ]
        print(f"🎯 High Quality Pin Bars (Strength 2+): {len(high_quality_pins)}")
        
        # Overall pattern summary
        total_patterns = self.processed_data[
            (self.processed_data['any_pin_bar'] == 1) | 
            (self.processed_data['any_engulfing'] == 1)
        ].shape[0]
        print(f"\n📈 Total Patterns Detected: {total_patterns}")
        print(f"📈 Pattern Coverage: {total_patterns/len(self.processed_data)*100:.1f}% of candles")
        
        return True

    def _detect_engulfing_patterns(self, df):
        """
        Add engulfing pattern detection to existing dataframe
        Private method called by process_training_data
        """
        df = df.copy()
        
        # Ensure we have the basic candle data
        if 'body_size' not in df.columns:
            df['body_size'] = abs(df['close'] - df['open'])
        
        # Determine candle colors
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        
        # Initialize engulfing columns
        df['bullish_engulfing'] = 0
        df['bearish_engulfing'] = 0
        
        # Bullish Engulfing: Today's green candle engulfs yesterday's red candle
        for i in range(1, len(df)):
            # Previous candle was red (bearish)
            prev_red = df.iloc[i-1]['is_red'] == 1
            # Current candle is green (bullish)  
            curr_green = df.iloc[i]['is_green'] == 1
            # Current candle's body engulfs previous candle's body
            engulfs_body = (df.iloc[i]['open'] < df.iloc[i-1]['close'] and 
                        df.iloc[i]['close'] > df.iloc[i-1]['open'])
            
            if prev_red and curr_green and engulfs_body:
                df.iloc[i, df.columns.get_loc('bullish_engulfing')] = 1
        
        # Bearish Engulfing: Today's red candle engulfs yesterday's green candle
        for i in range(1, len(df)):
            # Previous candle was green (bullish)
            prev_green = df.iloc[i-1]['is_green'] == 1
            # Current candle is red (bearish)
            curr_red = df.iloc[i]['is_red'] == 1
            # Current candle's body engulfs previous candle's body
            engulfs_body = (df.iloc[i]['open'] > df.iloc[i-1]['close'] and 
                        df.iloc[i]['close'] < df.iloc[i-1]['open'])
            
            if prev_green and curr_red and engulfs_body:
                df.iloc[i, df.columns.get_loc('bearish_engulfing')] = 1
        
        # Combined engulfing pattern
        df['any_engulfing'] = ((df['bullish_engulfing'] == 1) | 
                            (df['bearish_engulfing'] == 1)).astype(int)
        
        return df


    # STEP 4: Show Sample Results
    def show_pin_bar_samples(self, num_samples=5):
        """Display sample pin bars found"""
        if self.processed_data is None:
            print("❌ No processed data. Run process_training_data() first.")
            return
        
        pin_bars = self.processed_data[self.processed_data['any_pin_bar']==1]
        
        if len(pin_bars) == 0:
            print("❌ No pin bars found!")
            return
        
        sample = pin_bars[['time', 'open', 'high', 'low', 'close', 
                          'bullish_pin_bar', 'bearish_pin_bar', 'pin_bar_strength']].head(num_samples)
        
        print(f"\n📋 Sample Pin Bars (showing {len(sample)}):")
        print(sample.to_string(index=False))
    
    # STEP 5: Complete Pipeline
    def run_complete_pipeline(self):
        """Run the complete data collection and processing pipeline"""
        print("🚀 Starting Complete Pin Bar Training Pipeline")
        print("="*60)
        
        # Step 1: Setup MT5
        if not self.setup_mt5():
            return False
        
        # Step 2: Collect data
        if not self.collect_data():
            mt5.shutdown()
            return False
        
        # Step 3: Process data
        if not self.process_training_data():
            mt5.shutdown()
            return False
        
        # Step 4: Show samples
        self.show_pin_bar_samples()
        
        # Cleanup
        mt5.shutdown()
        print("\n✅ Pipeline completed successfully!")
        return True
    
    def create_success_targets(self, df, lookahead_candles=3, pip_target=10):
        """
        Create success labels for pin bar trades
        Success = price moves in predicted direction within lookahead_candles
        """
        df = df.copy()
        
        # Calculate pip value for EURUSD (4 decimal places)
        pip_value = 0.0001
        pip_target_price = pip_target * pip_value
        
        # Initialize success columns
        df['bullish_success'] = 0
        df['bearish_success'] = 0
        df['any_success'] = 0
        
        # For each candle, check success over next 2-4 candles
        for i in range(len(df) - lookahead_candles):
            current_close = df.iloc[i]['close']
            
            # Get the highest high and lowest low in the next lookahead_candles
            future_slice = df.iloc[i+1:i+1+lookahead_candles]
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            # Check bullish pin bar success
            if df.iloc[i]['bullish_pin_bar'] == 1:
                # Success if price moves up by pip_target within lookahead period
                if future_high >= (current_close + pip_target_price):
                    df.iloc[i, df.columns.get_loc('bullish_success')] = 1
                    df.iloc[i, df.columns.get_loc('any_success')] = 1
            
            # Check bearish pin bar success  
            if df.iloc[i]['bearish_pin_bar'] == 1:
                # Success if price moves down by pip_target within lookahead period
                if future_low <= (current_close - pip_target_price):
                    df.iloc[i, df.columns.get_loc('bearish_success')] = 1
                    df.iloc[i, df.columns.get_loc('any_success')] = 1
        
        return df

    def analyze_pin_bar_success_rates(self):
        """
        Analyze how successful pin bars are with current criteria
        """
        if self.processed_data is None:
            print("❌ No processed data available")
            return
        
        # Add success targets
        print("🎯 Creating success targets...")
        self.processed_data = self.create_success_targets(self.processed_data)
        
        # Analyze bullish pin bars
        bullish_pins = self.processed_data[self.processed_data['bullish_pin_bar'] == 1]
        bullish_successes = bullish_pins['bullish_success'].sum()
        bullish_total = len(bullish_pins)
        bullish_success_rate = (bullish_successes / bullish_total * 100) if bullish_total > 0 else 0
        
        # Analyze bearish pin bars
        bearish_pins = self.processed_data[self.processed_data['bearish_pin_bar'] == 1]
        bearish_successes = bearish_pins['bearish_success'].sum()
        bearish_total = len(bearish_pins)
        bearish_success_rate = (bearish_successes / bearish_total * 100) if bearish_total > 0 else 0
        
        # Overall success rate
        total_pins = self.processed_data[self.processed_data['any_pin_bar'] == 1]
        total_successes = total_pins['any_success'].sum()
        total_pin_count = len(total_pins)
        overall_success_rate = (total_successes / total_pin_count * 100) if total_pin_count > 0 else 0
        
        print(f"📊 Pin Bar Success Analysis (10 pips, 3 candles):")
        print(f"   Bullish Pin Bars: {bullish_successes}/{bullish_total} = {bullish_success_rate:.1f}% success")
        print(f"   Bearish Pin Bars: {bearish_successes}/{bearish_total} = {bearish_success_rate:.1f}% success")
        print(f"   Overall Success Rate: {total_successes}/{total_pin_count} = {overall_success_rate:.1f}%")
        
        # Analyze by pin bar strength
        print(f"\n📈 Success Rate by Pin Bar Strength:")
        for strength in range(4):
            strength_pins = total_pins[total_pins['pin_bar_strength'] == strength]
            if len(strength_pins) > 0:
                strength_successes = strength_pins['any_success'].sum()
                strength_rate = (strength_successes / len(strength_pins) * 100)
                print(f"   Strength {strength}: {strength_successes}/{len(strength_pins)} = {strength_rate:.1f}% success")
        
        return {
            'bullish_rate': bullish_success_rate,
            'bearish_rate': bearish_success_rate, 
            'overall_rate': overall_success_rate,
            'total_pins': total_pin_count,
            'successful_pins': total_successes
        }


    

# HOW TO USE THE STACKED CODE
if __name__ == "__main__":
    # Create trainer instance
    trainer = PinBarTrainer()
    
    # Option 1: Run complete pipeline (RECOMMENDED)
    trainer.run_complete_pipeline()
    
    # Option 2: Run step by step (if you want control)
    # trainer.setup_mt5()
    # trainer.collect_data()
    # trainer.process_training_data()
    # trainer.show_pin_bar_samples()
    # mt5.shutdown()
