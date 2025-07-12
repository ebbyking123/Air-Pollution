"""
Data validation and sample data creation script for Air Quality Prediction
"""
import pandas as pd
import numpy as np
import os
from config import *

def create_sample_data():
    """Create sample data based on the provided 20 records"""
    # Sample data from the problem statement
    sample_data = [
        [0, 51.491, -0.172, 301, 6, 3, 10, 14.5],
        [1, 39.386, 121.158, 254, 3, 21, 9, 34.5],
        [2, 51.459, 0.596, 301, 6, 3, 10, 10.5],
        [3, 35.299, -120.613, 145, 2, 14, 5, 15.5],
        [4, 29.927, 120.527, 221, 0, 14, 8, 54.5],
        [5, 43.676, -1.059, 285, 3, 11, 10, 15],
        [6, 29.961, 103.001, 221, 0, 14, 8, 17.5],
        [7, 26.177, 119.988, 53, 3, 3, 2, 16.5],
        [8, 46.382, -117.05, 151, 2, 2, 5, 7.9],
        [9, -23.717, 27.582, 150, 1, 0, 5, 29.291],
        [10, 35.094, 137.049, 267, 0, 10, 9, 10.5],
        [11, 50.432, 7.479, 29, 0, 22, 1, 26.123001],
        [12, 46.662, 124.916, 221, 0, 14, 8, 17.5],
        [13, -27.818, 153.248, 287, 5, 16, 10, 8.099999905],
        [14, 35.617, 139.694, 267, 0, 10, 9, 7.5],
        [15, 33.897, 118.367, 221, 0, 14, 8, 30.5],
        [16, 36.646, 109.368, 221, 0, 14, 8, 10.5],
        [17, 47.482, -120.518, 151, 2, 2, 5, 6.3],
        [18, 39.472, -87.412, 150, 1, 10, 5, 23.4],
        [19, 25.815, 29.508, 144, 2, 16, 5, 16.78],
        [20, 39.977, -75.203, 342, 2, 16, 12, 17.8]
    ]
    
    columns = ['id', 'latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month', 'pollution_value']
    
    # Create expanded sample data (multiply by 350 to get ~7000 records)
    expanded_data = []
    for i in range(350):
        for j, row in enumerate(sample_data):
            new_row = row.copy()
            new_row[0] = i * len(sample_data) + j  # New unique ID
            
            # Add some random variation to make it more realistic
            new_row[1] += np.random.normal(0, 0.1)  # latitude variation
            new_row[2] += np.random.normal(0, 0.1)  # longitude variation
            new_row[3] = np.random.randint(1, 366)  # random day of year
            new_row[4] = np.random.randint(0, 7)    # random day of week
            new_row[5] = np.random.randint(0, 24)   # random hour
            new_row[6] = np.random.randint(1, 13)   # random month
            new_row[7] += np.random.normal(0, 2)    # pollution value variation
            new_row[7] = max(0, new_row[7])         # ensure positive pollution
            
            expanded_data.append(new_row)
    
    return pd.DataFrame(expanded_data, columns=columns)

def create_test_data(train_df, test_size=2000):
    """Create test data (without pollution_value) from training data"""
    # Sample from training data
    test_sample = train_df.sample(n=test_size, random_state=42).copy()
    
    # Create new IDs for test data
    test_sample['id'] = range(len(test_sample))
    
    # Remove the target variable
    test_sample = test_sample.drop('pollution_value', axis=1)
    
    return test_sample

def validate_data_format(df, is_training=True):
    """Validate data format"""
    print(f"Validating {'training' if is_training else 'test'} data format...")
    
    required_cols = ['id', 'latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
    if is_training:
        required_cols.append('pollution_value')
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    
    # Check data types and ranges
    checks = [
        (df['latitude'].between(-90, 90).all(), "Latitude must be between -90 and 90"),
        (df['longitude'].between(-180, 180).all(), "Longitude must be between -180 and 180"),
        (df['day_of_year'].between(1, 365).all(), "Day of year must be between 1 and 365"),
        (df['day_of_week'].between(0, 6).all(), "Day of week must be between 0 and 6"),
        (df['hour'].between(0, 23).all(), "Hour must be between 0 and 23"),
        (df['month'].between(1, 12).all(), "Month must be between 1 and 12"),
    ]
    
    if is_training:
        checks.append((df['pollution_value'].notna().all(), "Pollution value cannot be null"))
        checks.append((df['pollution_value'].ge(0).all(), "Pollution value must be non-negative"))
    
    for check, message in checks:
        if not check:
            print(f"❌ {message}")
            return False
    
    print(f"✅ {'Training' if is_training else 'Test'} data format is valid")
    return True

def main():
    """Main data validation and creation function"""
    print("🔍 DATA VALIDATION AND CREATION TOOL")
    print("=" * 50)
    
    # Check if data files exist
    train_exists = os.path.exists(TRAIN_FILE)
    test_exists = os.path.exists(TEST_FILE)
    
    print(f"Train file exists: {train_exists}")
    print(f"Test file exists: {test_exists}")
    
    if not train_exists or not test_exists:
        print("\n📝 Creating sample data files...")
        
        # Create sample training data
        train_df = create_sample_data()
        print(f"Created sample training data: {train_df.shape}")
        
        # Create test data
        test_df = create_test_data(train_df, test_size=2000)
        print(f"Created sample test data: {test_df.shape}")
        
        # Save data files
        train_df.to_csv(TRAIN_FILE, index=False)
        test_df.to_csv(TEST_FILE, index=False)
        
        print(f"✅ Sample data files created:")
        print(f"   📄 {TRAIN_FILE}")
        print(f"   📄 {TEST_FILE}")
        
    else:
        print("\n📊 Validating existing data files...")
        
        # Load and validate existing data
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Validate formats
        train_valid = validate_data_format(train_df, is_training=True)
        test_valid = validate_data_format(test_df, is_training=False)
        
        if train_valid and test_valid:
            print("\n✅ All data files are valid and ready for training!")
        else:
            print("\n❌ Data validation failed. Please check your data files.")
            return False
    
    # Show data samples
    print("\n📋 DATA SAMPLES:")
    print("Training data (first 5 rows):")
    print(pd.read_csv(TRAIN_FILE).head())
    
    print("\nTest data (first 5 rows):")
    print(pd.read_csv(TEST_FILE).head())
    
    return True

if __name__ == "__main__":
    main()