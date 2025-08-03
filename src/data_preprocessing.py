"""
Data preprocessing pipeline for Moroccan barley yield dataset
"""

import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced cleaning with pH validation and imputation
    - Handles missing pH values by imputing with regional median
    - Validates and caps extreme pH values
    """
    # Initial cleaning  - remove rows with missing critical values
    critical_columns = ['yield_tons_ha', 'region', 'soil_ph']
    df = df.dropna(subset=[col for col in critical_columns if col != 'soil_ph'])

    # Handle pH values
    if 'soil_ph' in df.columns:
        # Step 1: Impute missing pH values with regional median
        df['soil_ph'] = df.groupby('region')['soil_ph'].transform(
            lambda x: x.fillna(x.median()))

        # Step 2: For regions with no median (all NaN), use overall median
        if df['soil_ph'].isna().any():
            overall_median = df['soil_ph'].median()
            df['soil_ph'] = df['soil_ph'].fillna(overall_median)

        # Step 3: Validate and cap pH values
        df['soil_ph'] = df['soil_ph'].clip(lower=5.5, upper=8.5)

        # Validate yield
        df = df[df['yield_tons_ha'] > 0]

    return df


def preprocess_data(
        input_path: Path = None,
        output_dir: Path = None
) -> pd.DataFrame:

    # Set default paths relative to project root
    project_root = get_project_root()
    if input_path is None:
        input_path = project_root / "data" / "raw" / "moroccan_barley_dataset.csv"
    if output_dir is None:
        output_dir = project_root / "data" / "processed"

    # Verify input file exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at: {input_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please ensure the file exists or provide the correct path."
        )

    print(f"Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded data with {len(df)} rows")

        # Your cleaning logic...
        df = df.dropna()
        print(f"After cleaning: {len(df)} rows remaining")

        # Save processed data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "processed.parquet"
        df.to_parquet(output_path)
        print(f"Saved processed data to {output_path}")

        return df

    except Exception as e:
        print(f"Error during processing: {str(e)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"Pipeline failed: {str(e)}", file=sys.stderr)
        sys.exit(1)