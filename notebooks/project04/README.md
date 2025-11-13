
# Project 04 — Kiruthikaa's Predicting Continuous Targets with Regression  (Titanic)
**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** November 12, 2025  

## Project Overview

This project explores two real-world datasets — **Titanic** and **California Housing** — to demonstrate how regression models can be used to predict continuous outcomes:

- **Titanic:** Predicting passenger **fare** based on demographic and travel features.
- **California Housing:** Predicting **median house prices** based on geographic and socioeconomic indicators.

The notebook showcases a full modeling pipeline including data cleaning, feature engineering, model comparison, and performance evaluation using metrics like **R²**, **RMSE**, and **MAE**. Visualizations are used throughout to support interpretability and model diagnostics.


## Files

- `ml04_regression.ipynb` — Main notebook containing both Titanic and California regression workflows.
- `README.md` — Repository-level summary.
- `docs/index.md` — Documentation for web or MkDocs hosting.
- (Optional) `requirements.txt` — Python dependencies.

## Dataset Information

### Titanic Dataset
- **Source:** `seaborn.load_dataset("titanic")`
- **Samples:** 891 passengers
- **Target:** `fare` (continuous)
- **Features Used:** `age`, `sex`, `family_size`, `pclass`, `embarked`

### California Housing Dataset
- **Source:** `sklearn.datasets.fetch_california_housing()`
- **Samples:** 20,640 block groups
- **Target:** `MedHouseVal` (median house value in $100,000s)
- **Features Used:** `MedInc`, `AveRooms`, `AveOccup`, `Latitude`, `Longitude`, etc.

## Key Steps in the Project

### Titanic Fare Prediction

1. **Data Cleaning & Feature Engineering**
   - Imputed missing values (`age`, `embarked`)
   - Created `family_size`, encoded `sex`, one-hot encoded `embarked`

2. **Exploratory Visualizations**
   - Histograms for `age`, `fare`, `family_size`
   - Count plots for `sex`, `pclass`, `embarked`
   - Scatter and box plots to explore relationships with fare

3. **Modeling**
   - Linear Regression across 4 feature cases
   - Ridge, ElasticNet, and Polynomial Regression (degree 3 and 8)
   - Visual comparison of predicted vs actual fares
   - R² bar chart and residual plots

4. **Findings**
   - Best simple model: `sex` alone (R² = 0.099)
   - Best overall model: Polynomial Regression (degree 3) with `age + family_size` (R² = 0.064)
   - All models underfit due to limited features; adding `pclass`, `embarked`, or cabin info could improve performance

### California Housing Price Prediction

1. **Data Loading & Visualization**
   - Histogram of target (`MedHouseVal`)
   - Scatterplot of predicted vs actual prices
   - Residual plot to assess variance and bias

2. **Modeling**
   - Linear Regression on full feature set
   - Performance: R² = 0.608, RMSE ≈ $71,900, MAE ≈ $52,700

3. **Findings**
   - Stronger fit than Titanic due to richer feature set
   - Geographic and income features are key drivers of housing prices

## How to Run

Clone the repository:

```bash
git clone https://github.com/<kiruthikaa2512>/regression-project.git
cd regression-project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook ml04_regression.ipynb
```

Run all cells sequentially to execute both regression workflows.

## Learning Outcomes

- Applied regression modeling to real-world datasets with continuous targets.
- Compared linear, regularized, and polynomial models.
- Used visual diagnostics to interpret model fit and residuals.
- Demonstrated reproducible workflows with clear feature engineering and evaluation.
- Highlighted the importance of feature richness and non-linearity in predictive modeling.

