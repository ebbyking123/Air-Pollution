"""
Utility functions for Air Quality Prediction ML Model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_submission(test_ids, predictions, filename):
    """Create submission file"""
    submission_df = pd.DataFrame({
        'id': test_ids,
        'pollution_value': predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"✅ Submission file created: {filename}")
    print(f"   Shape: {submission_df.shape}")
    print(f"   Sample predictions:")
    print(submission_df.head())
    
    return submission_df

def analyze_data(train_df, test_df):
    """Perform comprehensive data analysis"""
    print("📊 DATA ANALYSIS")
    print("-" * 40)
    
    # Basic statistics
    print("Training Data Overview:")
    print(f"  Shape: {train_df.shape}")
    print(f"  Missing values: {train_df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {train_df.duplicated().sum()}")
    
    print("\nTest Data Overview:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Missing values: {test_df.isnull().sum().sum()}")
    
    # Target variable statistics
    print(f"\nTarget Variable (pollution_value) Statistics:")
    print(f"  Mean: {train_df['pollution_value'].mean():.2f}")
    print(f"  Median: {train_df['pollution_value'].median():.2f}")
    print(f"  Std: {train_df['pollution_value'].std():.2f}")
    print(f"  Min: {train_df['pollution_value'].min():.2f}")
    print(f"  Max: {train_df['pollution_value'].max():.2f}")
    print(f"  Skewness: {train_df['pollution_value'].skew():.2f}")
    
    # Feature statistics
    print("\nFeature Statistics:")
    for col in ['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'day_of_year']:
        if col in train_df.columns:
            print(f"  {col}: min={train_df[col].min():.2f}, max={train_df[col].max():.2f}, "
                  f"unique={train_df[col].nunique()}")
    
    # Correlation analysis
    print("\nCorrelation with target:")
    correlations = train_df.corr()['pollution_value'].sort_values(ascending=False)
    print(correlations[correlations.index != 'pollution_value'])
    
    return train_df.describe(), test_df.describe()

def plot_results(y_true, y_pred_ensemble, individual_predictions, base_results):
    """Create comprehensive result plots"""
    print("📈 Creating result plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Air Quality Prediction Model Results', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs Actual scatter plot
    axes[0, 0].scatter(y_true, y_pred_ensemble, alpha=0.6, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Ensemble: Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² score
    r2 = 1 - np.sum((y_true - y_pred_ensemble)**2) / np.sum((y_true - np.mean(y_true))**2)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Residual plot
    residuals = y_true - y_pred_ensemble
    axes[0, 1].scatter(y_pred_ensemble, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Residuals')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Model performance comparison
    if base_results:
        model_names = []
        rmse_scores = []
        custom_scores = []
        
        for name, results in base_results.items():
            if results:
                model_names.append(name)
                rmse_scores.append(results['val_rmse'])
                custom_scores.append(results['custom_metric'])
        
        axes[1, 0].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Model RMSE Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Individual model predictions comparison
    if individual_predictions:
        for i, (name, pred) in enumerate(individual_predictions.items()):
            if i < 3:  # Show only first 3 models
                axes[1, 1].scatter(y_true, pred, alpha=0.5, s=15, label=name)
        
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Individual Model Predictions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature importance (if available)
    axes[1, 2].text(0.5, 0.5, 'Feature Importance\n(Available after training)', 
                    ha='center', va='center', fontsize=12, 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Results plots saved as 'model_results.png'")

def create_interactive_plots(train_df, predictions=None):
    """Create interactive plots using Plotly"""
    print("Creating interactive plots...")
    
    # 1. Spatial distribution of pollution
    fig = px.scatter_mapbox(
        train_df, 
        lat="latitude", 
        lon="longitude", 
        color="pollution_value",
        size="pollution_value",
        hover_data=["hour", "day_of_week", "month"],
        color_continuous_scale="Viridis",
        title="Spatial Distribution of Air Pollution",
        zoom=3
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html("spatial_pollution_map.html")
    
    # 2. Temporal patterns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Hourly Pattern", "Daily Pattern", "Monthly Pattern", "Yearly Pattern")
    )
    
    # Hourly pattern
    hourly_avg = train_df.groupby('hour')['pollution_value'].mean()
    fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values, 
                           mode='lines+markers', name='Hourly Avg'), row=1, col=1)
    
    # Daily pattern
    daily_avg = train_df.groupby('day_of_week')['pollution_value'].mean()
    fig.add_trace(go.Scatter(x=daily_avg.index, y=daily_avg.values, 
                           mode='lines+markers', name='Daily Avg'), row=1, col=2)
    
    # Monthly pattern
    monthly_avg = train_df.groupby('month')['pollution_value'].mean()
    fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, 
                           mode='lines+markers', name='Monthly Avg'), row=2, col=1)
    
    # Yearly pattern
    yearly_avg = train_df.groupby('day_of_year')['pollution_value'].mean()
    fig.add_trace(go.Scatter(x=yearly_avg.index, y=yearly_avg.values, 
                           mode='lines', name='Yearly Avg'), row=2, col=2)
    
    fig.update_layout(title_text="Temporal Patterns in Air Pollution", showlegend=False)
    fig.write_html("temporal_patterns.html")
    
    print("✅ Interactive plots created: spatial_pollution_map.html, temporal_patterns.html")

def validate_submission(submission_file):
    """Validate submission file format"""
    try:
        df = pd.read_csv(submission_file)
        
        # Check columns
        required_cols = ['id', 'pollution_value']
        if not all(col in df.columns for col in required_cols):
            print("❌ Missing required columns")
            return False
        
        # Check for missing values
        if df['pollution_value'].isnull().any():
            print("❌ Missing values in pollution_value column")
            return False
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['pollution_value']):
            print("❌ pollution_value must be numeric")
            return False
        
        print("✅ Submission file validation passed")
        print(f"   Shape: {df.shape}")
        print(f"   Prediction range: [{df['pollution_value'].min():.2f}, {df['pollution_value'].max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False