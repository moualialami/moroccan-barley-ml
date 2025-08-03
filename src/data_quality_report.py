"""
Generates data quality reports
"""
import pandas as pd
from pathlib import Path


def generate_quality_report():
    df = pd.read_parquet(Path("../data/processed/processed.parquet"))

    report = {
        "n_rows": len(df),
        "pH_range": (df['soil_ph'].min(), df['soil_ph'].max()),
        "yield_range": (df['yield_tons_ha'].min(), df['yield_tons_ha'].max()),
        "missing_values": df.isnull().sum().to_dict()
    }

    print("Data Quality Report:")
    print(f"- Rows: {report['n_rows']}")
    print(f"- pH Range: {report['pH_range'][0]:.1f} - {report['pH_range'][1]:.1f}")
    print("- Missing Values:", report['missing_values'])


if __name__ == "__main__":
    generate_quality_report()