"""
Train Decision Tree model for barley yield prediction
"""

import pandas as pd
import joblib
from pathlib import Path
import sys
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


def train_model(
        data_path: Path = None,
        model_dir: Path = None,
        random_state: int = 42
) -> DecisionTreeRegressor:
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
        print("Training Decision Tree model...")
        model = DecisionTreeRegressor(
            max_depth=5,  # Controls tree depth to prevent overfitting
            min_samples_split=10,  # Minimum samples required to split a node
            min_samples_leaf=5,  # Minimum samples required at a leaf node
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\nModel Performance:")
        print(f"RMSE: {rmse:.4f} t/ha")
        print(f"R² Score: {r2:.4f}")

        # Save model
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "decision_tree_model.joblib"
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")

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