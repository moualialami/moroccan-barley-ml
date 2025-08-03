"""
Feature Engineering Pipeline for Moroccan Barley Yield Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Configuration
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed.parquet"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "reports"/ "features"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load processed data with validation"""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    return pd.read_parquet(PROCESSED_DATA_PATH)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw data
    Returns DataFrame with original + engineered features
    """
    # 1. Soil Nutrient Ratios
    df['np_ratio'] = df['nitrogen_content_percent'] / (df['phosphorus_content_ppm'] + 1e-6)
    df['soil_fertility_index'] = (
            df['nitrogen_content_percent'] *
            df['phosphorus_content_ppm'] *
            df['potassium_content_ppm']
    )

    # 2. Climate Features
    df['temp_rain_balance'] = df['growing_season_temp_c'] / (df['rainfall_mm'] + 1)
    df['aridity_index'] = df['rainfall_mm'] / (df['growing_season_temp_c'] + 10)

    # 3. Interaction Terms
    df['ph_organic_interaction'] = df['soil_ph'] * df['organic_matter_percent']

    return df


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Apply preprocessing pipeline:
    - Scaling to numerical features
    - One-hot encoding for categoricals
    Returns transformed DataFrame and fitted preprocessor
    """
    # Identify feature types
    numerical_features = [
        'growing_season_temp_c', 'rainfall_mm', 'soil_ph',
        'organic_matter_percent', 'np_ratio', 'soil_fertility_index',
        'temp_rain_balance', 'aridity_index', 'ph_organic_interaction'
    ]

    categorical_features = ['region', 'soil_type', 'barley_variety']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit and transform
    features = preprocessor.fit_transform(df)

    # Get feature names
    num_feature_names = numerical_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])

    # Create DataFrame with named features
    features_df = pd.DataFrame(features, columns=all_feature_names)
    features_df['yield_tons_ha'] = df['yield_tons_ha'].values

    return features_df, preprocessor


def save_artifacts(features_df: pd.DataFrame, preprocessor: ColumnTransformer):
    """Save engineered features and preprocessing pipeline"""
    features_df.to_parquet(FEATURES_DATA_PATH)
    joblib.dump(preprocessor, MODELS_DIR / "feature_preprocessor.joblib")
    print(f"✅ Saved features to {FEATURES_DATA_PATH}")
    print(f"✅ Saved preprocessor to {MODELS_DIR / 'feature_preprocessor.joblib'}")


def analyze_features(df: pd.DataFrame):
    """Generate feature analysis visualizations"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Numerical feature distributions
        plt.figure(figsize=(10, 6))
        sns.histplot(df['soil_fertility_index'], bins=30, kde=True)
        plt.title('Soil Fertility Index Distribution')
        plt.savefig(PLOTS_DIR / "soil_fertility_dist.png")
        plt.close()

        # Feature-target relationships
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='ph_organic_interaction', y='yield_tons_ha', data=df)
        plt.title('pH-Organic Matter Interaction vs Yield')
        plt.savefig(PLOTS_DIR / "ph_organic_vs_yield.png")
        plt.close()

    except ImportError:
        print("⚠️ Visualization libraries not installed - skipping plots")


def main():
    print("🚀 Starting feature engineering pipeline...")

    # Load and validate data
    df = load_data()
    print(f"📊 Loaded data with shape: {df.shape}")

    # Feature creation
    df = create_features(df)
    print("🔧 Engineered new features")

    # Preprocessing
    features_df, preprocessor = preprocess_features(df)
    print(f"🧹 Preprocessed features. New shape: {features_df.shape}")

    # Save artifacts
    save_artifacts(features_df, preprocessor)

    # Analysis (optional)
    analyze_features(df)
    print(f"📈 Feature analysis plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()