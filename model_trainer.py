"""
Model training and evaluation module for Air Quality Prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_models = {}
        self.predictions = {}
        
    def initialize_models(self):
        """Initialize all models"""
        self.models = {
            'random_forest': RandomForestRegressor(
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.config.RANDOM_STATE),
            'lasso': Lasso(random_state=self.config.RANDOM_STATE)
        }
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val):
        """Evaluate a single model"""
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Calculate custom metric: exp(-RMSE/100)
        custom_metric = np.exp(-val_rmse/100)
        
        return {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'custom_metric': custom_metric
        }
    
    def train_base_models(self, X_train, y_train, X_val, y_val):
        """Train all base models"""
        print("Training base models...")
        
        self.initialize_models()
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val)
                results[name] = metrics
                
                print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
                print(f"  Val RMSE: {metrics['val_rmse']:.4f}")
                print(f"  Val MAE: {metrics['val_mae']:.4f}")
                print(f"  Val R²: {metrics['val_r2']:.4f}")
                print(f"  Custom Metric: {metrics['custom_metric']:.6f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                results[name] = None
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\nTuning hyperparameters for {model_name}...")
        
        if model_name not in self.config.MODELS_CONFIG:
            print(f"No hyperparameter config found for {model_name}")
            return None
        
        # Get base model
        if model_name == 'random_forest':
            base_model = RandomForestRegressor(random_state=self.config.RANDOM_STATE, n_jobs=-1)
        elif model_name == 'xgboost':
            base_model = xgb.XGBRegressor(random_state=self.config.RANDOM_STATE, n_jobs=-1)
        elif model_name == 'lightgbm':
            base_model = lgb.LGBMRegressor(random_state=self.config.RANDOM_STATE, n_jobs=-1, verbose=-1)
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        param_grid = self.config.MODELS_CONFIG[model_name]
        
        # Use a smaller grid for faster tuning
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            }
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=3,  # Reduced CV for faster tuning
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
        
        self.best_models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create ensemble of best models"""
        print("\nCreating ensemble model...")
        
        # Train best models if not already trained
        if not self.best_models:
            print("Training models for ensemble...")
            
            # Use base models with good default parameters
            self.best_models['random_forest'] = RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=2,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            )
            self.best_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=self.config.RANDOM_STATE, n_jobs=-1
            )
            self.best_models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.1, num_leaves=50,
                random_state=self.config.RANDOM_STATE, n_jobs=-1, verbose=-1
            )
            
            # Train all models
            for name, model in self.best_models.items():
                print(f"Training {name} for ensemble...")
                model.fit(X_train, y_train)
        
        # Make predictions with each model
        ensemble_predictions = {}
        
        for name, model in self.best_models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X_val)
                ensemble_predictions[name] = pred
        
        # Create weighted ensemble
        weights = self.config.ENSEMBLE_WEIGHTS
        ensemble_pred = np.zeros_like(list(ensemble_predictions.values())[0])
        
        for name, pred in ensemble_predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        # Evaluate ensemble
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        ensemble_custom = np.exp(-ensemble_rmse/100)
        
        print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
        print(f"Ensemble Custom Metric: {ensemble_custom:.6f}")
        
        return ensemble_pred, ensemble_predictions
    
    def make_predictions(self, X_test):
        """Make predictions on test data"""
        print("\nMaking predictions on test data...")
        
        predictions = {}
        
        # Individual model predictions
        for name, model in self.best_models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X_test)
                predictions[name] = pred
                print(f"{name} predictions shape: {pred.shape}")
        
        # Ensemble prediction
        weights = self.config.ENSEMBLE_WEIGHTS
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def save_models(self, filepath_prefix='model'):
        """Save trained models"""
        print(f"\nSaving models...")
        
        for name, model in self.best_models.items():
            filename = f"{filepath_prefix}_{name}.joblib"
            joblib.dump(model, filename)
            print(f"Saved {name} model to {filename}")
    
    def load_models(self, filepath_prefix='model'):
        """Load trained models"""
        print(f"\nLoading models...")
        
        for name in ['random_forest', 'xgboost', 'lightgbm']:
            try:
                filename = f"{filepath_prefix}_{name}.joblib"
                model = joblib.load(filename)
                self.best_models[name] = model
                print(f"Loaded {name} model from {filename}")
            except FileNotFoundError:
                print(f"Could not find {filename}")