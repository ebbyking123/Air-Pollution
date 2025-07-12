"""
Main execution script for Air Quality Prediction ML Model
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from config import *
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from utils import create_submission, plot_results, analyze_data

def main():
    """Main execution function"""
    print("=" * 60)
    print("🌬️  AIR QUALITY PREDICTION ML MODEL")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize components
    print("\n1. Initializing components...")
    data_processor = DataProcessor(config=sys.modules[__name__])
    model_trainer = ModelTrainer(config=sys.modules[__name__])
    
    # Load data
    print("\n2. Loading data...")
    if not data_processor.load_data(TRAIN_FILE, TEST_FILE):
        print("❌ Failed to load data. Please ensure train.csv and test.csv are in the data/ directory.")
        return
    
    # Analyze data
    print("\n3. Analyzing data...")
    analyze_data(data_processor.train_df, data_processor.test_df)
    
    # Prepare data
    print("\n4. Preparing data...")
    X_train, X_val, y_train, y_val, X_test = data_processor.prepare_data(scale_features=True)
    
    # Train base models
    print("\n5. Training base models...")
    base_results = model_trainer.train_base_models(X_train, y_train, X_val, y_val)
    
    # Hyperparameter tuning for best performing models
    print("\n6. Hyperparameter tuning...")
    best_model_name = max(base_results.keys(), 
                         key=lambda x: base_results[x]['custom_metric'] if base_results[x] else 0)
    print(f"Best base model: {best_model_name}")
    
    # Tune top 2 models
    top_models = sorted(base_results.items(), 
                       key=lambda x: x[1]['custom_metric'] if x[1] else 0, 
                       reverse=True)[:2]
    
    for model_name, _ in top_models:
        if model_name in ['random_forest', 'xgboost', 'lightgbm']:
            model_trainer.hyperparameter_tuning(X_train, y_train, model_name)
    
    # Create ensemble
    print("\n7. Creating ensemble model...")
    ensemble_val_pred, individual_val_preds = model_trainer.create_ensemble(
        X_train, y_train, X_val, y_val
    )
    
    # Make final predictions
    print("\n8. Making final predictions...")
    test_predictions = model_trainer.make_predictions(X_test)
    
    # Create submission file
    print("\n9. Creating submission file...")
    test_ids = data_processor.get_test_ids()
    create_submission(test_ids, test_predictions['ensemble'], SUBMISSION_FILE)
    
    # Plot results
    print("\n10. Generating result plots...")
    plot_results(y_val, ensemble_val_pred, individual_val_preds, base_results)
    
    # Save models
    print("\n11. Saving models...")
    model_trainer.save_models()
    
    print("\n" + "=" * 60)
    print("✅ AIR QUALITY PREDICTION MODEL COMPLETED!")
    print("=" * 60)
    print(f"Completed at: {datetime.now()}")
    print(f"📄 Submission file created: {SUBMISSION_FILE}")
    print(f"📊 Models saved and plots generated")
    print(f"📈 Final validation RMSE: {np.sqrt(np.mean((y_val - ensemble_val_pred)**2)):.4f}")
    print(f"🎯 Custom metric: {np.exp(-np.sqrt(np.mean((y_val - ensemble_val_pred)**2))/100):.6f}")

if __name__ == "__main__":
    main()