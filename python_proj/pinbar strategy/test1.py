# Then run these commands step by step:
from pin_bar_trainer import PinBarTrainer
trainer = PinBarTrainer()
# Import the fixed functions
from chart_visualizer import quick_training_chart, quick_testing_chart, show_pattern_legend


# Step 2: Setup MT5 connection
print("🔧 Setting up MT5...")
trainer.setup_mt5()



# Step 3: Collect EURUSD 15M data
print("📡 Collecting data...")
trainer.collect_data(2000)



# Step 4: Process pin bar detection on training data
print("🎯 Processing pin bars...")
trainer.process_training_data() 


show_pattern_legend()

# Now visualize with both pattern types
from chart_visualizer import quick_training_chart
quick_training_chart(trainer, candles=100)









# Create and analyze success targets
print("🎯 Analyzing pin bar success rates...")
success_stats = trainer.analyze_pin_bar_success_rates()

# Show some successful vs failed pin bars
successful_pins = trainer.processed_data[
    (trainer.processed_data['any_pin_bar'] == 1) & 
    (trainer.processed_data['any_success'] == 1)
]

failed_pins = trainer.processed_data[
    (trainer.processed_data['any_pin_bar'] == 1) & 
    (trainer.processed_data['any_success'] == 0)
]

print(f"\n📋 Sample Successful Pin Bars:")
if len(successful_pins) > 0:
    sample_success = successful_pins[['time', 'open', 'high', 'low', 'close', 
                                     'bullish_pin_bar', 'bearish_pin_bar', 
                                     'pin_bar_strength']].head(3)
    print(sample_success.to_string(index=False))

print(f"\n📋 Sample Failed Pin Bars:")
if len(failed_pins) > 0:
    sample_failed = failed_pins[['time', 'open', 'high', 'low', 'close', 
                                'bullish_pin_bar', 'bearish_pin_bar', 
                                'pin_bar_strength']].head(3)
    print(sample_failed.to_string(index=False))




# # Check what we collected
# print(f"Training data: {len(trainer.train_data)} candles")
# print(f"Testing data: {len(trainer.test_data)} candles")
# print("\nFirst 3 training candles:")
# print(trainer.train_data[['time', 'open', 'high', 'low', 'close']].head(3))