# src/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def run_eda(data_path: Path = Path("../data/processed/processed.parquet")):
    """Run exploratory data analysis and save plots."""
    # Create plots directory
    reports_dir = Path("../reports/eda")
    reports_dir.mkdir(parents=True, exist_ok=True)  # Creates all parent dirs

    # Load data
    df = pd.read_parquet(data_path)
    print("✅ Data loaded. Shape:", df.shape)

    # 1. Target Variable Analysis
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["yield_tons_ha"], bins=30, kde=True, ax=ax[0])
    sns.boxplot(y=df["yield_tons_ha"], ax=ax[1])
    plt.savefig(reports_dir / "yield_distribution.png")
    plt.close()

    # 2. Feature Distributions (Numerical Only)
    num_cols = ["growing_season_temp_c", "rainfall_mm", "soil_ph", "organic_matter_percent"]
    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], bins=25, kde=True)
        plt.savefig(reports_dir / f"{col}_distribution.png")
        plt.close()

    # 3. Correlation Analysis (Numerical Only)
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    if not numerical_df.empty:
        corr = numerical_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title('Numerical Features Correlation')
        plt.savefig(reports_dir / "correlation_matrix.png")
        plt.close()

        print("📊 Numerical correlations computed")
    else:
        print("⚠️ No numerical columns for correlation analysis")

    print(f"✅ EDA plots saved to {reports_dir}/")


if __name__ == "__main__":
    run_eda()