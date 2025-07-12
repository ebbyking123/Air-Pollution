"""
Configuration settings for the Air Quality Prediction Model
"""
import os

# Data paths
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = "submission.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering settings
CYCLICAL_FEATURES = ['hour', 'day_of_week', 'month', 'day_of_year']
SPATIAL_FEATURES = ['latitude', 'longitude']
TARGET_COLUMN = 'pollution_value'
ID_COLUMN = 'id'

# Model hyperparameters
MODELS_CONFIG = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0]
    }
}

# Ensemble settings
ENSEMBLE_WEIGHTS = {
    'random_forest': 0.3,
    'xgboost': 0.4,
    'lightgbm': 0.3
}