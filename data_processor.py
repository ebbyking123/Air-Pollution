"""
Data processing and feature engineering module for Air Quality Prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None
        self.spatial_clusterer = None
        
    def load_data(self, train_path, test_path):
        """Load training and test data"""
        print("Loading data...")
        
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            # Clean data - handle NaN values
            print("Cleaning data...")
            
            # Fill NaN values with appropriate defaults
            numeric_columns = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
            for col in numeric_columns:
                if col in self.train_df.columns:
                    self.train_df[col] = self.train_df[col].fillna(self.train_df[col].median())
                if col in self.test_df.columns:
                    self.test_df[col] = self.test_df[col].fillna(self.test_df[col].median())
            
            # Handle pollution_value NaN in training data
            if 'pollution_value' in self.train_df.columns:
                self.train_df['pollution_value'] = self.train_df['pollution_value'].fillna(self.train_df['pollution_value'].median())
            
            print(f"Training data shape: {self.train_df.shape}")
            print(f"Test data shape: {self.test_df.shape}")
            
            # Basic data info
            print("\nTraining data info:")
            print(self.train_df.info())
            print("\nTraining data statistics:")
            print(self.train_df.describe())
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_cyclical_features(self, df):
        """Create cyclical encoding for temporal features"""
        df_copy = df.copy()
        
        # Hour (0-23)
        df_copy['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6)
        df_copy['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df_copy['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (1-12)
        df_copy['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # Day of year (1-365)
        df_copy['doy_sin'] = np.sin(2 * np.pi * (df['day_of_year'] - 1) / 365)
        df_copy['doy_cos'] = np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
        
        return df_copy
    
    def create_spatial_features(self, df):
        """Create spatial features from coordinates"""
        df_copy = df.copy()
        
        # Handle NaN values with imputation instead of dropping
        from sklearn.impute import SimpleImputer
        spatial_imputer = SimpleImputer(strategy='median')
        
        # Prepare spatial data for clustering
        spatial_cols = ['latitude', 'longitude']
        spatial_data = df_copy[spatial_cols].values
        
        # Impute missing values
        spatial_data_clean = spatial_imputer.fit_transform(spatial_data)
        df_copy[spatial_cols] = spatial_data_clean
        
        # Create spatial clusters
        if self.spatial_clusterer is None:
            self.spatial_clusterer = KMeans(n_clusters=50, random_state=self.config.RANDOM_STATE)
            self.spatial_clusterer.fit(spatial_data_clean)
        
        # Add cluster labels
        spatial_clusters = self.spatial_clusterer.predict(spatial_data_clean)
        df_copy['spatial_cluster'] = spatial_clusters
        
        # Add distance from centroid
        centroids = self.spatial_clusterer.cluster_centers_
        cluster_centroids = centroids[spatial_clusters]
        
        df_copy['dist_from_centroid'] = np.sqrt(
            (df_copy['latitude'] - cluster_centroids[:, 0])**2 + 
            (df_copy['longitude'] - cluster_centroids[:, 1])**2
        )
        
        # Add coordinate interactions
        df_copy['lat_lon_interaction'] = df_copy['latitude'] * df_copy['longitude']
        df_copy['lat_squared'] = df_copy['latitude']**2
        df_copy['lon_squared'] = df_copy['longitude']**2
        
        return df_copy
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df_copy = df.copy()
        
        # Time-space interactions
        df_copy['lat_hour'] = df_copy['latitude'] * df_copy['hour']
        df_copy['lon_hour'] = df_copy['longitude'] * df_copy['hour']
        df_copy['lat_month'] = df_copy['latitude'] * df_copy['month']
        df_copy['lon_month'] = df_copy['longitude'] * df_copy['month']
        
        # Temporal interactions
        df_copy['hour_dow'] = df_copy['hour'] * df_copy['day_of_week']
        df_copy['month_dow'] = df_copy['month'] * df_copy['day_of_week']
        
        return df_copy
    
    def engineer_features(self, df, is_training=True):
        """Complete feature engineering pipeline"""
        print("Engineering features...")
        
        df_processed = df.copy()
        
        # Create cyclical features
        df_processed = self.create_cyclical_features(df_processed)
        
        # Create spatial features
        df_processed = self.create_spatial_features(df_processed)
        
        # Create interaction features
        df_processed = self.create_interaction_features(df_processed)
        
        # Select feature columns (exclude id and target)
        feature_cols = [col for col in df_processed.columns 
                       if col not in [self.config.ID_COLUMN, self.config.TARGET_COLUMN]]
        
        if is_training:
            self.feature_names = feature_cols
        
        print(f"Total features created: {len(feature_cols)}")
        print(f"Features: {feature_cols}")
        
        return df_processed
    
    def prepare_data(self, scale_features=True):
        """Prepare data for model training"""
        print("Preparing data for training...")
        
        # Engineer features
        train_processed = self.engineer_features(self.train_df, is_training=True)
        test_processed = self.engineer_features(self.test_df, is_training=False)
        
        # Extract features and target
        X = train_processed[self.feature_names].values
        y = train_processed[self.config.TARGET_COLUMN].values
        X_test = test_processed[self.feature_names].values
        
        # Scale features if requested
        if scale_features:
            print("Scaling features...")
            X = self.scaler.fit_transform(X)
            X_test = self.scaler.transform(X_test)
        
        # Split training data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=None
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test
    
    def get_test_ids(self):
        """Get test data IDs for submission"""
        return self.test_df[self.config.ID_COLUMN].values