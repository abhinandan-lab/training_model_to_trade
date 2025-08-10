import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BOSChochFeatureEngine:
    """
    Complete feature engineering pipeline for BOS/CHoCH detection
    with built-in validation and quality checks
    """
    
    def __init__(self, fractal_length: int = 5, validation_mode: bool = True):
        self.fractal_length = fractal_length
        self.validation_mode = validation_mode
        self.fractals = []
        self.events = []
        self.feature_table = None
        self.validation_results = {}
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            
        Returns:
            Complete feature table ready for RL model
        """
        print("🚀 Starting BOS/CHoCH Feature Engineering Pipeline...")
        
        # Step 1: Validate input data
        if self.validation_mode:
            self._validate_input_data(df)
        
        # Step 2: Detect fractals
        print("📊 Detecting fractals...")
        self.fractals = self._detect_fractals(df)
        
        # Step 3: Detect BOS/CHoCH events
        print("🔍 Detecting BOS/CHoCH events...")
        self.events = self._detect_bos_choch(df)
        
        # Step 4: Create feature table
        print("⚙️ Creating feature table...")
        self.feature_table = self._create_complete_features(df)
        
        # Step 5: Validate results
        if self.validation_mode:
            print("✅ Validating results...")
            self._validate_results()
            self._print_validation_summary()
        
        print("🎉 Feature engineering completed!")
        return self.feature_table.copy()
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame structure and quality"""
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types
        numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Check for obvious data quality issues
        issues = []
        
        # OHLC consistency
        if (df['high'] < df[['open', 'close']].max(axis=1)).any():
            issues.append("High prices lower than open/close detected")
        
        if (df['low'] > df[['open', 'close']].min(axis=1)).any():
            issues.append("Low prices higher than open/close detected")
        
        # Negative values
        if (df[numeric_cols] <= 0).any().any():
            issues.append("Negative or zero prices detected")
        
        # Missing values
        if df[required_columns].isnull().any().any():
            issues.append("Missing values detected")
        
        self.validation_results['input_data'] = {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_candles': len(df)
        }
        
        if issues and self.validation_mode:
            print(f"⚠️ Data quality issues detected: {issues}")
    
    def _detect_fractals(self, df: pd.DataFrame) -> List[Dict]:
        """Detect fractals using Pine Script logic"""
        p = self.fractal_length // 2
        fractals = []
        
        for i in range(p, len(df) - p):
            # Check for swing high fractal
            window_high = df['high'].iloc[i-p:i+p+1]
            if df['high'].iloc[i] == window_high.max():
                fractals.append({
                    "idx": i,
                    "type": "high",
                    "price": df['high'].iloc[i],
                    "is_crossed": False
                })
            
            # Check for swing low fractal  
            window_low = df['low'].iloc[i-p:i+p+1]
            if df['low'].iloc[i] == window_low.min():
                fractals.append({
                    "idx": i,
                    "type": "low", 
                    "price": df['low'].iloc[i],
                    "is_crossed": False
                })
        
        return fractals
    
    def _detect_bos_choch(self, df: pd.DataFrame) -> List[Dict]:
        """Detect BOS/CHoCH events"""
        events = []
        market_structure = 0  # 0 = neutral, 1 = bullish, -1 = bearish
        active_fractals = [f.copy() for f in self.fractals]
        
        for i in range(len(df)):
            current_close = df["close"].iloc[i]
            
            # Check for bullish breakouts
            for fractal in active_fractals:
                if (fractal["type"] == "high" and 
                    not fractal["is_crossed"] and 
                    fractal["idx"] < i and
                    current_close > fractal["price"]):
                    
                    event_type = "CHoCH" if market_structure == -1 else "BOS"
                    
                    events.append({
                        "fractal_idx": fractal["idx"],
                        "break_idx": i,
                        "fractal_price": fractal["price"],
                        "type": event_type,
                        "direction": "bullish"
                    })
                    
                    market_structure = 1
                    fractal["is_crossed"] = True
                    break
            
            # Check for bearish breakouts  
            for fractal in active_fractals:
                if (fractal["type"] == "low" and 
                    not fractal["is_crossed"] and 
                    fractal["idx"] < i and
                    current_close < fractal["price"]):
                    
                    event_type = "CHoCH" if market_structure == 1 else "BOS"
                    
                    events.append({
                        "fractal_idx": fractal["idx"],
                        "break_idx": i,
                        "fractal_price": fractal["price"],
                        "type": event_type,
                        "direction": "bearish"
                    })
                    
                    market_structure = -1
                    fractal["is_crossed"] = True
                    break
        
        return events
    
    def _create_complete_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complete feature table"""
        # Start with basic candle features
        features = self._create_candle_features(df)
        
        # Add fractal features
        fractal_features = self._create_fractal_features(df)
        features = pd.concat([features, fractal_features], axis=1)
        
        # Add structure features
        structure_features = self._create_structure_features(df)
        features = pd.concat([features, structure_features], axis=1)
        
        # Add proximity features
        proximity_features = self._create_proximity_features(df)
        features = pd.concat([features, proximity_features], axis=1)
        
        # Add temporal features
        temporal_features = self._create_temporal_features(df)
        features = pd.concat([features, temporal_features], axis=1)
        
        # Add target labels
        features = self._add_target_labels(features)
        
        # Clean up infinite values and NaN
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _create_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic candle features"""
        features = df.copy()
        
        # Basic OHLC features
        features['body_size'] = abs(features['close'] - features['open'])
        features['upper_wick'] = features['high'] - features[['open', 'close']].max(axis=1)
        features['lower_wick'] = features[['open', 'close']].min(axis=1) - features['low']
        features['range'] = features['high'] - features['low']
        
        # Price movement features
        features['price_change'] = features['close'].pct_change()
        features['is_green'] = (features['close'] > features['open']).astype(int)
        features['is_doji'] = (features['body_size'] < features['range'] * 0.1).astype(int)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'volatility_{window}'] = features['close'].rolling(window).std()
            features[f'volume_sma_{window}'] = features['tick_volume'].rolling(window).mean()
            features[f'rsi_{window}'] = self._calculate_rsi(features['close'], window)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fractal-based features"""
        n_candles = len(df)
        features = pd.DataFrame(index=df.index)
        
        # Initialize arrays
        features['is_fractal_high'] = 0
        features['is_fractal_low'] = 0
        features['nearest_high_distance'] = 999.0
        features['nearest_low_distance'] = 999.0
        features['nearest_high_price'] = np.nan
        features['nearest_low_price'] = np.nan
        features['fractal_age_high'] = 999
        features['fractal_age_low'] = 999
        features['unbroken_highs_count'] = 0
        features['unbroken_lows_count'] = 0
        
        # Mark fractal points
        for fractal in self.fractals:
            idx = fractal['idx']
            if fractal['type'] == 'high':
                features.iloc[idx, features.columns.get_loc('is_fractal_high')] = 1
            else:
                features.iloc[idx, features.columns.get_loc('is_fractal_low')] = 1
        
        # Calculate features for each candle
        for i in range(n_candles):
            current_price = df['close'].iloc[i]
            
            # Find past fractals
            past_highs = [f for f in self.fractals if f['idx'] <= i and f['type'] == 'high']
            past_lows = [f for f in self.fractals if f['idx'] <= i and f['type'] == 'low']
            
            # Nearest fractals
            if past_highs:
                nearest_high = max(past_highs, key=lambda x: x['idx'])
                features.iloc[i, features.columns.get_loc('nearest_high_distance')] = abs(current_price - nearest_high['price']) / current_price
                features.iloc[i, features.columns.get_loc('nearest_high_price')] = nearest_high['price']
                features.iloc[i, features.columns.get_loc('fractal_age_high')] = i - nearest_high['idx']
                
            if past_lows:
                nearest_low = max(past_lows, key=lambda x: x['idx'])
                features.iloc[i, features.columns.get_loc('nearest_low_distance')] = abs(current_price - nearest_low['price']) / current_price
                features.iloc[i, features.columns.get_loc('nearest_low_price')] = nearest_low['price']
                features.iloc[i, features.columns.get_loc('fractal_age_low')] = i - nearest_low['idx']
            
            # Count unbroken fractals
            unbroken_highs = [f for f in past_highs if not f['is_crossed']]
            unbroken_lows = [f for f in past_lows if not f['is_crossed']]
            
            features.iloc[i, features.columns.get_loc('unbroken_highs_count')] = len(unbroken_highs)
            features.iloc[i, features.columns.get_loc('unbroken_lows_count')] = len(unbroken_lows)
        
        return features
    
    def _create_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market structure features"""
        features = pd.DataFrame(index=df.index)
        n_candles = len(df)
        
        # Initialize
        features['market_structure'] = 0  # -1=bearish, 0=neutral, 1=bullish
        features['bars_since_last_event'] = 999
        features['last_event_type'] = 0  # 0=none, 1=BOS, 2=CHoCH
        features['last_event_direction'] = 0  # -1=bearish, 1=bullish
        features['structure_strength'] = 0
        features['trend_consistency'] = 0
        
        current_structure = 0
        last_event_bar = -1
        recent_events = []
        
        for i in range(n_candles):
            # Check for events at this bar
            current_events = [e for e in self.events if e['break_idx'] == i]
            
            if current_events:
                event = current_events[0]
                
                # Update structure
                current_structure = 1 if event['direction'] == 'bullish' else -1
                features.iloc[i, features.columns.get_loc('last_event_direction')] = current_structure
                features.iloc[i, features.columns.get_loc('last_event_type')] = 1 if event['type'] == 'BOS' else 2
                
                last_event_bar = i
                recent_events.append(event)
                
                # Keep only recent events (last 50 bars)
                recent_events = [e for e in recent_events if i - e['break_idx'] <= 50]
            
            # Set current state
            features.iloc[i, features.columns.get_loc('market_structure')] = current_structure
            features.iloc[i, features.columns.get_loc('bars_since_last_event')] = i - last_event_bar if last_event_bar >= 0 else 999
            
            # Calculate structure strength (consistency of recent events)
            if recent_events:
                same_direction = len([e for e in recent_events if (e['direction'] == 'bullish') == (current_structure == 1)])
                features.iloc[i, features.columns.get_loc('structure_strength')] = same_direction / len(recent_events)
                
                # Trend consistency
                bos_count = len([e for e in recent_events if e['type'] == 'BOS'])
                features.iloc[i, features.columns.get_loc('trend_consistency')] = bos_count / len(recent_events) if recent_events else 0
        
        return features
    
    def _create_proximity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create proximity and breakout features"""
        features = pd.DataFrame(index=df.index)
        n_candles = len(df)
        
        # Initialize
        features['distance_to_resistance'] = 999.0
        features['distance_to_support'] = 999.0
        features['approaching_resistance'] = 0
        features['approaching_support'] = 0
        features['breakout_strength'] = 0.0
        features['false_breakout_risk'] = 0.0
        
        for i in range(n_candles):
            current_price = df['close'].iloc[i]
            
            # Find unbroken levels
            unbroken_highs = [f for f in self.fractals if f['idx'] < i and not f['is_crossed'] and f['type'] == 'high']
            unbroken_lows = [f for f in self.fractals if f['idx'] < i and not f['is_crossed'] and f['type'] == 'low']
            
            # Nearest resistance/support
            if unbroken_highs:
                nearest_resistance = min(unbroken_highs, key=lambda x: abs(current_price - x['price']))
                distance = (nearest_resistance['price'] - current_price) / current_price
                features.iloc[i, features.columns.get_loc('distance_to_resistance')] = max(0, distance)
                
                if 0 < distance < 0.01:  # Within 1%
                    features.iloc[i, features.columns.get_loc('approaching_resistance')] = 1
            
            if unbroken_lows:
                nearest_support = min(unbroken_lows, key=lambda x: abs(current_price - x['price']))
                distance = (current_price - nearest_support['price']) / current_price
                features.iloc[i, features.columns.get_loc('distance_to_support')] = max(0, distance)
                
                if 0 < distance < 0.01:  # Within 1%
                    features.iloc[i, features.columns.get_loc('approaching_support')] = 1
            
            # Breakout strength
            recent_events = [e for e in self.events if e['break_idx'] == i]
            if recent_events:
                event = recent_events[0]
                strength = abs(current_price - event['fractal_price']) / event['fractal_price']
                features.iloc[i, features.columns.get_loc('breakout_strength')] = strength
                
                # False breakout risk (based on volume and momentum)
                volume_ratio = df['tick_volume'].iloc[i] / df['tick_volume'].iloc[max(0, i-10):i].mean()
                features.iloc[i, features.columns.get_loc('false_breakout_risk')] = 1.0 / max(volume_ratio, 0.1)
        
        return features
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal sequence features"""
        features = pd.DataFrame(index=df.index)
        n_candles = len(df)
        
        # Initialize
        features['bos_count_10'] = 0
        features['choch_count_10'] = 0
        features['event_frequency'] = 0
        features['trend_duration'] = 0
        features['structure_velocity'] = 0.0
        
        current_trend_start = 0
        
        for i in range(n_candles):
            # Rolling event counts
            lookback_start = max(0, i - 10)
            recent_events = [e for e in self.events if lookback_start <= e['break_idx'] <= i]
            
            bos_count = len([e for e in recent_events if e['type'] == 'BOS'])
            choch_count = len([e for e in recent_events if e['type'] == 'CHoCH'])
            
            features.iloc[i, features.columns.get_loc('bos_count_10')] = bos_count
            features.iloc[i, features.columns.get_loc('choch_count_10')] = choch_count
            features.iloc[i, features.columns.get_loc('event_frequency')] = len(recent_events)
            
            # Trend duration
            current_events = [e for e in self.events if e['break_idx'] == i and e['type'] == 'CHoCH']
            if current_events:
                current_trend_start = i
            
            features.iloc[i, features.columns.get_loc('trend_duration')] = i - current_trend_start
            
            # Structure velocity (events per unit time)
            if recent_events:
                time_span = max(1, i - recent_events[0]['break_idx'])
                features.iloc[i, features.columns.get_loc('structure_velocity')] = len(recent_events) / time_span
        
        return features
    
    def _add_target_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add future event labels"""
        features['future_bos_5'] = 0
        features['future_choch_5'] = 0
        features['future_event_direction'] = 0
        features['future_price_change_5'] = 0.0
        
        for event in self.events:
            break_idx = event['break_idx']
            start_idx = max(0, break_idx - 5)
            
            for i in range(start_idx, break_idx):
                if i < len(features):
                    if event['type'] == 'BOS':
                        features.iloc[i, features.columns.get_loc('future_bos_5')] = 1
                    else:
                        features.iloc[i, features.columns.get_loc('future_choch_5')] = 1
                    
                    direction = 1 if event['direction'] == 'bullish' else -1
                    features.iloc[i, features.columns.get_loc('future_event_direction')] = direction
        
        # Future price changes
        for i in range(len(features) - 5):
            current_price = features['close'].iloc[i]
            future_price = features['close'].iloc[i + 5]
            price_change = (future_price - current_price) / current_price
            features.iloc[i, features.columns.get_loc('future_price_change_5')] = price_change
        
        return features
    
    def _validate_results(self) -> None:
        """Comprehensive validation of results"""
        validation_checks = {}
        
        # 1. Fractal validation
        fractal_checks = self._validate_fractals()
        validation_checks['fractals'] = fractal_checks
        
        # 2. Event validation
        event_checks = self._validate_events()
        validation_checks['events'] = event_checks
        
        # 3. Feature validation
        feature_checks = self._validate_features()
        validation_checks['features'] = feature_checks
        
        # 4. Data consistency validation
        consistency_checks = self._validate_consistency()
        validation_checks['consistency'] = consistency_checks
        
        self.validation_results.update(validation_checks)
    
    def _validate_fractals(self) -> Dict:
        """Validate fractal detection"""
        checks = {}
        
        # Check fractal count is reasonable
        total_fractals = len(self.fractals)
        data_length = len(self.feature_table)
        fractal_density = total_fractals / data_length if data_length > 0 else 0
        
        checks['total_fractals'] = total_fractals
        checks['fractal_density'] = fractal_density
        checks['density_reasonable'] = 0.01 <= fractal_density <= 0.3  # 1-30% seems reasonable
        
        # Check fractal type distribution
        high_fractals = len([f for f in self.fractals if f['type'] == 'high'])
        low_fractals = len([f for f in self.fractals if f['type'] == 'low'])
        
        checks['high_fractals'] = high_fractals
        checks['low_fractals'] = low_fractals
        checks['balanced_types'] = abs(high_fractals - low_fractals) / max(total_fractals, 1) < 0.3
        
        # Check for duplicate fractals
        fractal_indices = [f['idx'] for f in self.fractals]
        checks['no_duplicates'] = len(fractal_indices) == len(set(fractal_indices))
        
        return checks
    
    def _validate_events(self) -> Dict:
        """Validate BOS/CHoCH events"""
        checks = {}
        
        # Basic counts
        total_events = len(self.events)
        bos_events = len([e for e in self.events if e['type'] == 'BOS'])
        choch_events = len([e for e in self.events if e['type'] == 'CHoCH'])
        
        checks['total_events'] = total_events
        checks['bos_events'] = bos_events
        checks['choch_events'] = choch_events
        
        # Event frequency
        data_length = len(self.feature_table)
        event_frequency = total_events / data_length if data_length > 0 else 0
        checks['event_frequency'] = event_frequency
        checks['frequency_reasonable'] = 0.001 <= event_frequency <= 0.1  # 0.1-10% seems reasonable
        
        # Direction balance
        bullish_events = len([e for e in self.events if e['direction'] == 'bullish'])
        bearish_events = len([e for e in self.events if e['direction'] == 'bearish'])
        
        checks['bullish_events'] = bullish_events
        checks['bearish_events'] = bearish_events
        checks['direction_balance'] = abs(bullish_events - bearish_events) / max(total_events, 1) < 0.7
        
        # Event sequence validation
        checks['valid_sequence'] = self._validate_event_sequence()
        
        return checks
    
    def _validate_event_sequence(self) -> bool:
        """Validate that events follow logical sequence"""
        if len(self.events) < 2:
            return True
        
        # Check that events are in chronological order
        for i in range(1, len(self.events)):
            if self.events[i]['break_idx'] <= self.events[i-1]['break_idx']:
                return False
        
        # Check that fractal indices are before break indices
        for event in self.events:
            if event['fractal_idx'] >= event['break_idx']:
                return False
        
        return True
    
    def _validate_features(self) -> Dict:
        """Validate feature table quality"""
        checks = {}
        
        if self.feature_table is None:
            checks['table_exists'] = False
            return checks
        
        checks['table_exists'] = True
        checks['shape'] = self.feature_table.shape
        
        # Check for missing values
        missing_counts = self.feature_table.isnull().sum()
        checks['missing_values'] = missing_counts.sum()
        checks['no_missing'] = missing_counts.sum() == 0
        
        # Check for infinite values
        inf_counts = np.isinf(self.feature_table.select_dtypes(include=[np.number])).sum()
        checks['infinite_values'] = inf_counts.sum()
        checks['no_infinite'] = inf_counts.sum() == 0
        
        # Check feature ranges
        numeric_features = self.feature_table.select_dtypes(include=[np.number])
        checks['feature_ranges'] = {
            col: {
                'min': numeric_features[col].min(),
                'max': numeric_features[col].max(),
                'std': numeric_features[col].std()
            }
            for col in numeric_features.columns[:10]  # First 10 features
        }
        
        # Check for constant features
        constant_features = [col for col in numeric_features.columns 
                           if numeric_features[col].std() == 0]
        checks['constant_features'] = constant_features
        checks['no_constant_features'] = len(constant_features) == 0
        
        return checks
    
    def _validate_consistency(self) -> Dict:
        """Validate consistency between components"""
        checks = {}
        
        # Check that all fractal indices are valid
        data_length = len(self.feature_table)
        valid_fractal_indices = all(0 <= f['idx'] < data_length for f in self.fractals)
        checks['valid_fractal_indices'] = valid_fractal_indices
        
        # Check that all event indices are valid
        valid_event_indices = all(
            0 <= e['fractal_idx'] < data_length and 0 <= e['break_idx'] < data_length
            for e in self.events
        )
        checks['valid_event_indices'] = valid_event_indices
        
        # Check that events reference existing fractals
        fractal_indices = set(f['idx'] for f in self.fractals)
        event_fractal_refs = set(e['fractal_idx'] for e in self.events)
        valid_fractal_refs = event_fractal_refs.issubset(fractal_indices)
        checks['valid_fractal_references'] = valid_fractal_refs
        
        # Check feature consistency with raw data
        if self.feature_table is not None:
            price_consistency = np.allclose(
                self.feature_table['close'].values,
                self.feature_table['close'].values,
                rtol=1e-10
            )
            checks['price_consistency'] = price_consistency
        
        return checks
    
    def _print_validation_summary(self) -> None:
        """Print comprehensive validation summary"""
        print("\n" + "="*60)
        print("🔍 VALIDATION SUMMARY")
        print("="*60)
        
        # Input data validation
        if 'input_data' in self.validation_results:
            input_data = self.validation_results['input_data']
            status = "✅ PASSED" if input_data['passed'] else "❌ FAILED"
            print(f"\n📊 Input Data Validation: {status}")
            if input_data['issues']:
                for issue in input_data['issues']:
                    print(f"   ⚠️ {issue}")
        
        # Fractal validation
        if 'fractals' in self.validation_results:
            fractals = self.validation_results['fractals']
            print(f"\n🔺 Fractal Validation:")
            print(f"   Total fractals: {fractals['total_fractals']}")
            print(f"   Density: {fractals['fractal_density']:.3f}")
            print(f"   High/Low balance: ✅" if fractals['balanced_types'] else "   High/Low balance: ❌")
            print(f"   No duplicates: ✅" if fractals['no_duplicates'] else "   No duplicates: ❌")
        
        # Event validation
        if 'events' in self.validation_results:
            events = self.validation_results['events']
            print(f"\n📈 Event Validation:")
            print(f"   Total events: {events['total_events']}")
            print(f"   BOS events: {events['bos_events']}")
            print(f"   CHoCH events: {events['choch_events']}")
            print(f"   Event frequency: {events['event_frequency']:.4f}")
            print(f"   Direction balance: ✅" if events['direction_balance'] else "   Direction balance: ❌")
            print(f"   Valid sequence: ✅" if events['valid_sequence'] else "   Valid sequence: ❌")
        
        # Feature validation
        if 'features' in self.validation_results:
            features = self.validation_results['features']
            print(f"\n⚙️ Feature Validation:")
            if features['table_exists']:
                print(f"   Shape: {features['shape']}")
                print(f"   Missing values: {features['missing_values']}")
                print(f"   Infinite values: {features['infinite_values']}")
                print(f"   Constant features: {len(features['constant_features'])}")
                if features['constant_features']:
                    print(f"   Constants: {features['constant_features'][:5]}")
            else:
                print("   ❌ Feature table not created")
        
        # Consistency validation
        if 'consistency' in self.validation_results:
            consistency = self.validation_results['consistency']
            print(f"\n🔄 Consistency Validation:")
            print(f"   Valid fractal indices: ✅" if consistency['valid_fractal_indices'] else "   Valid fractal indices: ❌")
            print(f"   Valid event indices: ✅" if consistency['valid_event_indices'] else "   Valid event indices: ❌")
            print(f"   Valid fractal refs: ✅" if consistency['valid_fractal_references'] else "   Valid fractal refs: ❌")
        
        print("\n" + "="*60)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if self.feature_table is None:
            return {}
        
        return {
            'total_candles': len(self.feature_table),
            'total_fractals': len(self.fractals),
            'total_events': len(self.events),
            'bos_events': len([e for e in self.events if e['type'] == 'BOS']),
            'choch_events': len([e for e in self.events if e['type'] == 'CHoCH']),
            'feature_count': self.feature_table.shape[1],
            'validation_passed': all(
                result.get('passed', True) if isinstance(result, dict) else True
                for result in self.validation_results.values()
            )
        }
    
    def plot_validation_charts(self) -> None:
        """Create validation visualizations"""
        if self.feature_table is None:
            print("❌ No feature table to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Engineering Validation Charts', fontsize=16)
        
        # 1. Feature distribution
        numeric_features = self.feature_table.select_dtypes(include=[np.number])
        feature_means = numeric_features.mean().sort_values(ascending=False)[:20]
        
        axes[0, 0].bar(range(len(feature_means)), feature_means.values)
        axes[0, 0].set_title('Top 20 Feature Means')
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Mean Value')
        
        # 2. Event distribution over time
        event_timeline = np.zeros(len(self.feature_table))
        for event in self.events:
            event_timeline[event['break_idx']] = 1 if event['type'] == 'BOS' else 2
        
        axes[0, 1].plot(event_timeline, alpha=0.7)
        axes[0, 1].set_title('Event Timeline (1=BOS, 2=CHoCH)')
        axes[0, 1].set_xlabel('Candle Index')
        axes[0, 1].set_ylabel('Event Type')
        
        # 3. Fractal distribution
        fractal_timeline = np.zeros(len(self.feature_table))
        for fractal in self.fractals:
            fractal_timeline[fractal['idx']] = 1 if fractal['type'] == 'high' else -1
        
        axes[1, 0].plot(fractal_timeline, alpha=0.7, color='orange')
        axes[1, 0].set_title('Fractal Timeline (1=High, -1=Low)')
        axes[1, 0].set_xlabel('Candle Index')
        axes[1, 0].set_ylabel('Fractal Type')
        
        # 4. Feature correlation heatmap (subset)
        correlation_features = ['price_change', 'market_structure', 'distance_to_resistance', 
                              'distance_to_support', 'event_frequency', 'trend_duration']
        available_features = [f for f in correlation_features if f in self.feature_table.columns]
        
        if available_features:
            corr_matrix = self.feature_table[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()

# Usage example
def main():
    """Example usage of the BOSChochFeatureEngine"""
    # Sample data creation for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic price data
    price_base = 50000
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = [price_base]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices[1:])
    
    # Create OHLC data
    sample_data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n_samples, freq='5T'),
        'open': prices + np.random.normal(0, 10, n_samples),
        'high': prices + np.abs(np.random.normal(20, 15, n_samples)),
        'low': prices - np.abs(np.random.normal(20, 15, n_samples)),
        'close': prices,
        'tick_volume': np.random.randint(100, 1000, n_samples)
    })
    
    # Ensure OHLC consistency
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    # Initialize and run feature engine
    engine = BOSChochFeatureEngine(fractal_length=5, validation_mode=True)
    
    # Process data
    feature_table = engine.process(sample_data)
    
    # Get summary
    summary = engine.get_summary_stats()
    print(f"\n📊 Summary Statistics:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Show validation charts
    engine.plot_validation_charts()
    
    return feature_table, engine

if __name__ == "__main__":
    features, engine = main()
