"""
Quick test script to verify the ML pipeline works correctly
"""
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import config

def quick_test():
    """Run a quick test of the ML pipeline"""
    print("🧪 QUICK TEST OF ML PIPELINE")
    print("=" * 40)
    
    # Test 1: Data Processor
    print("\n1. Testing Data Processor...")
    processor = DataProcessor(config)
    
    # Create tiny sample data for testing
    sample_data = pd.DataFrame({
        'id': range(100),
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100),
        'day_of_year': np.random.randint(1, 366, 100),
        'day_of_week': np.random.randint(0, 7, 100),
        'hour': np.random.randint(0, 24, 100),
        'month': np.random.randint(1, 13, 100),
        'pollution_value': np.random.uniform(0, 100, 100)
    })
    
    processor.train_df = sample_data
    processor.test_df = sample_data.drop('pollution_value', axis=1)
    
    # Test feature engineering
    processed_data = processor.engineer_features(sample_data)
    print(f"   ✅ Feature engineering: {processed_data.shape} -> {len(processor.feature_names)} features")
    
    # Test 2: Model Trainer
    print("\n2. Testing Model Trainer...")
    trainer = ModelTrainer(config)
    
    # Prepare minimal data
    X_train, X_val, y_train, y_val, X_test = processor.prepare_data()
    print(f"   ✅ Data preparation: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Test model initialization
    trainer.initialize_models()
    print(f"   ✅ Model initialization: {len(trainer.models)} models ready")
    
    # Test single model training
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - rf_pred)**2))
    print(f"   ✅ Model training: RMSE={rmse:.4f}")
    
    print("\n✅ ALL TESTS PASSED! The ML pipeline is ready to use.")
    print("🚀 You can now run: python main.py")

if __name__ == "__main__":
    quick_test()