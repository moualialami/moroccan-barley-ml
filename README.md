# 🌾 Barley Yield Prediction Using Machine Learning (Synthetic Educational Project)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project demonstrates how machine learning can be applied to predict **barley yields** using synthetic but realistic agronomic data inspired by Moroccan farming systems.

> ⚠️ **Disclaimer:** The data used here is fully synthetic and intended for educational and demonstration purposes only. Do not use it for operational agricultural decisions.

---

## 📌 Project Summary

This repository provides a full ML pipeline to simulate precision agriculture applications, specifically for barley yield prediction, using realistic input variables:

- **Environmental conditions** (temperature, rainfall, soil pH)
- **Soil nutrients** (N, P, K, organic matter)
- **Farm management** (fertilizer rates, seed density, crop variety)
- **Geographical context** (regions, soil types)

🎯 **Goal**: Showcase how data science can support data-driven decision-making in dryland agriculture.

---

## ✨ Key Features

- End-to-end pipeline: data cleaning → feature engineering → ML modeling
- Comparative analysis of 4 ML models
- Explainable AI: Feature importance analysis & interpretability
- Modular codebase ready for real-world adaptation
- Structured for clarity and reproducibility

---

## 📊 Dataset Overview

Synthetic dataset with **1500 observations** and realistic ranges for Moroccan agronomy.

### Categorical Features

| Feature         | Description                     | Sample Values                            |
|----------------|----------------------------------|------------------------------------------|
| `region`        | Barley production region         | Chaouia, Fès-Boulemane, Gharb, etc.       |
| `soil_type`     | Soil classification              | Vertisol, Clay, Loam, Sandy               |
| `barley_variety`| Barley cultivar used             | Khnata, Assiya, Amira, Chifaa, Massine    |

### Numerical Features

| Feature                   | Unit   | Description                              | Range         |
|---------------------------|--------|------------------------------------------|---------------|
| `growing_season_temp_c`   | °C     | Avg. temperature during season           | 16–24         |
| `rainfall_mm`             | mm     | Total seasonal rainfall                  | 150–600       |
| `soil_ph`                 | pH     | Soil acidity/alkalinity                  | 5.5–8.5       |
| `organic_matter_percent`  | %      | Soil organic matter                      | 0.8–3.5       |
| `nitrogen_content_percent`| %      | Soil nitrogen                            | 0.08–0.25     |
| `phosphorus_content_ppm`  | ppm    | Available phosphorus                     | 8–35          |
| `potassium_content_ppm`   | ppm    | Potassium level                          | 80–220        |
| `seed_rate_kg_ha`         | kg/ha  | Sowing density                           | 140–180       |
| `fertilizer_npk_kg_ha`    | kg/ha  | NPK fertilizer rate                      | 60–280        |

### Target

| Feature         | Unit | Description            |
|-----------------|------|------------------------|
| `yield_tons_ha` | t/ha | Barley grain yield     |

---

## 📈 Results Snapshot

- Random Forest outperformed baseline and linear models (R² ≈ 0.81)
- Most influential factors: **rainfall**, **soil pH**, and **fertilizer NPK**
- Yield variance linked to regional and varietal differences (~15%)

---

## 🛠 Technical Stack

- Python 3.8+
- pandas, numpy, matplotlib
- scikit-learn, xgboost
- Jupyter notebooks (EDA)
- Modular Python scripts (`src/`)
- Tested with `pytest` (optional)

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/moualialami/moroccan-barley-ml.git
cd moroccan-barley-ml

# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_gradient_boosting.py
python src/train_decision_tree.py
python src/train_random_forest.py
python src/train_linear_regression.py
```

---

## Project Structure

```
moroccan-barley-ml/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_quality_report.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_gradient_boosting.py
│   ├── train_random_forest.py
│   ├── train_linear_regression.py
│   ├── train_decision_tree.py
│   └── evaluate.py
├── models/
├── reports/
├── tests/
│   └── test_data_quality.py
├── requirements.txt
└── README.md
```


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- Inspired by Morocco’s agroecological diversity

📬 Contact

- Mohammed OUALI ALAMI
- Email: ouali.alami.mohammed@gmail.com
- GitHub: moualialami

