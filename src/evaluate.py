"""
Evaluate trained barley yield prediction model with feature importance
"""

import pandas as pd
import joblib
from pathlib import Path
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import json
import sklearn
import matplotlib.pyplot as plt
import numpy as np

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent

def calculate_rmse(y_true, y_pred):

    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        mse = mean_squared_error(y_true, y_pred)
        return mse ** 0.5

def plot_feature_importance(importance, features, output_dir: Path):

    plt.figure(figsize=(10, 6))
    idx = np.argsort(importance)
    plt.barh(range(len(features)), importance[idx], color='skyblue')
    plt.yticks(range(len(features)), np.array(features)[idx])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance for Barley Yield Prediction')
    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'feature_importance.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def evaluate_model(
    data_path: Path = None,
    model_path: Path = None,
    metrics_path: Path = None,
    importance_path: Path = None  # New parameter
) -> dict:

    # Set default paths
    project_root = get_project_root()
    if data_path is None:
        data_path = project_root / "data" / "processed" / "processed.parquet"
    if model_path is None:
        model_path = project_root / "models" / "rf_model.joblib"
    if metrics_path is None:
        metrics_path = project_root / "models" / "metrics.json"
    if importance_path is None:  # New
        importance_path = project_root / "reports" / "figures"

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
        feature_names = X.columns.tolist()

        # Make predictions
        predictions = model.predict(X)

        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y, predictions),
            'rmse': calculate_rmse(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'n_samples': len(y),
            'sklearn_version': sklearn.__version__
        }

        # Permutation Importance ---
        print("\nCalculating feature importance...")
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            scoring='neg_mean_squared_error'
        )

        importance = result.importances_mean
        metrics['feature_importance'] = dict(zip(feature_names, importance.tolist()))


        plot_path = plot_feature_importance(importance, feature_names, importance_path)
        print(f"Feature importance plot saved to: {plot_path}")

        # Print top
        top_features = sorted(zip(feature_names, importance),
                            key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 Important Features:")
        for feat, imp in top_features:
            print(f"- {feat}: {imp:.4f}")

        # Print results
        print("\nEvaluation Results:")
        print(f"- R² Score: {metrics['r2_score']:.3f}")
        print(f"- RMSE: {metrics['rmse']:.3f} tons/ha")
        print(f"- MAE: {metrics['mae']:.3f} tons/ha")
        print(f"- Samples: {metrics['n_samples']}")

        # Save metrics
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

        return metrics

    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)
