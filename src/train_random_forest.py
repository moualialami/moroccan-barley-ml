"""
Train machine learning models for barley yield prediction
"""

import pandas as pd
import joblib
from pathlib import Path
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent

def train_model(
    data_path: Path = None,
    model_dir: Path = None,
    random_state: int = 42
) -> RandomForestRegressor:

    # Set default paths relative to project root
    project_root = get_project_root()
    if data_path is None:
        data_path = project_root / "data" / "processed" / "processed.parquet"
    if model_dir is None:
        model_dir = project_root / "models"

    # Verify input file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at: {data_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please run data_preprocessing.py first to generate the processed data."
        )

    print(f"Loading processed data from: {data_path}")
    try:
        df = pd.read_parquet(data_path)
        print(f"Successfully loaded {len(df)} records")

        # Prepare features
        X = pd.get_dummies(df.drop('yield_tons_ha', axis=1))
        y = df['yield_tons_ha']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Train model
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            verbose=1
        )
        model.fit(X_train, y_train)

        # Save model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "rf_model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        return model

    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)