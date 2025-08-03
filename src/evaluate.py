"""
Evaluate trained barley yield prediction model
"""

import pandas as pd
import joblib
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import sklearn

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent

def calculate_rmse(y_true, y_pred):
    """Version-safe RMSE calculation"""
    try:
        # Try new version with squared parameter
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Fallback to manual calculation for older versions
        mse = mean_squared_error(y_true, y_pred)
        return mse ** 0.5

def evaluate_model(
    data_path: Path = None,
    model_path: Path = None,
    metrics_path: Path = None
) -> dict:

    # Set default paths
    project_root = get_project_root()
    if data_path is None:
        data_path = project_root / "data" / "processed" / "processed.parquet"
    if model_path is None:
        model_path = project_root / "models" / "rf_model.joblib"
    if metrics_path is None:
        metrics_path = project_root / "models" / "metrics.json"

    # Verify files exist
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found at {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"🔍 Loading data from: {data_path}")
    print(f"🔍 Loading model from: {model_path}")
    print(f"ℹ️ Using scikit-learn version: {sklearn.__version__}")

    try:
        # Load data and model
        df = pd.read_parquet(data_path)
        model = joblib.load(model_path)

        # Prepare features
        X = pd.get_dummies(df.drop('yield_tons_ha', axis=1))
        y = df['yield_tons_ha']

        # Make predictions
        predictions = model.predict(X)

        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y, predictions),
            'rmse': calculate_rmse(y, predictions),  # Using safe calculation
            'mae': mean_absolute_error(y, predictions),
            'n_samples': len(y),
            'sklearn_version': sklearn.__version__
        }

        # Print results
        print("\n📊 Evaluation Results:")
        print(f"- R² Score: {metrics['r2_score']:.3f}")
        print(f"- RMSE: {metrics['rmse']:.3f} tons/ha")
        print(f"- MAE: {metrics['mae']:.3f} tons/ha")
        print(f"- Samples: {metrics['n_samples']}")

        # Save metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to: {metrics_path}")

        return metrics

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)