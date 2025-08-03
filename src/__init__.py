from pathlib import Path

# Package version
__version__ = "1.0.0"

# Define important paths as package constants
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"

# Import main functions from your existing scripts
from .data_preprocessing import preprocess_data
from .data_quality_report import generate_quality_report
from .eda import run_eda
from .evaluate import evaluate_model
from .feature_engineering import create_features, preprocess_features,save_artifacts, analyze_features, main
from .train_decision_tree import train_model
from .train_gradient_boosting import train_model
from .train_random_forest import train_model
from .train_linear_regression import train_model



__all__ = [
    'preprocess_data',
    'train_model',
    'evaluate_model',
    'DATA_DIR',
    'MODEL_DIR'
]