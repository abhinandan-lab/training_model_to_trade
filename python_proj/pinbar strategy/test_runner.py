# test_runner.py
from pin_bar_trainer import PinBarTrainer

# Create trainer instance
trainer = PinBarTrainer()

# Step 1: Setup MT5
print("🔧 Step 1: Setting up MT5...")
if trainer.setup_mt5():
    print("✅ MT5 setup complete")
    
    # Step 2: Collect data
    print("\n📡 Step 2: Collecting data...")
    if trainer.collect_data(2000):
        print("✅ Data collection complete")
        
        # Step 3: Process data with pin bar detection
        print("\n🎯 Step 3: Processing pin bar detection...")
        if trainer.process_training_data():
            print("✅ Pin bar detection complete")
            
            # Step 4: Show samples
            print("\n📋 Step 4: Showing pin bar samples...")
            trainer.show_pin_bar_samples(10)
            
            print("\n✅ All steps completed successfully!")
        else:
            print("❌ Pin bar processing failed")
    else:
        print("❌ Data collection failed")
else:
    print("❌ MT5 setup failed")
