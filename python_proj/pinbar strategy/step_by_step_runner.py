# step_by_step_runner.py
from pin_bar_trainer import PinBarTrainer

def main():
    trainer = PinBarTrainer()
    
    # Step 1
    print("🔧 Step 1: MT5 Setup")
    success = trainer.setup_mt5()
    print(f"Result: {success}")
    input("Press Enter to continue to Step 2...")
    
    if not success:
        return
    
    # Step 2
    print("\n📡 Step 2: Data Collection")
    success = trainer.collect_data(2000)
    print(f"Training data shape: {trainer.train_data.shape if trainer.train_data is not None else 'Failed'}")
    input("Press Enter to continue to Step 3...")
    
    if not success:
        return
    
    # Step 3
    print("\n🎯 Step 3: Pin Bar Detection")
    success = trainer.process_training_data()
    if success and trainer.processed_data is not None:
        pin_count = trainer.processed_data['any_pin_bar'].sum()
        print(f"Pin bars found: {pin_count}")
    input("Press Enter to continue to Step 4...")
    
    if not success:
        return
    
    # Step 4
    print("\n📋 Step 4: Sample Results")
    trainer.show_pin_bar_samples(5)
    
    # Optional: Inspect data
    print(f"\n🔍 Data available for inspection:")
    print(f"- trainer.train_data: {len(trainer.train_data)} rows")
    print(f"- trainer.test_data: {len(trainer.test_data)} rows") 
    print(f"- trainer.processed_data: {len(trainer.processed_data)} rows")
    
    input("Press Enter to finish...")

if __name__ == "__main__":
    main()
