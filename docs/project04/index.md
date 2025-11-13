# Project 04 — Regression with Titanic Fare + California Housing  
**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** November 12, 2025  

**Notebook**  
You can view the full Jupyter notebook for this project here:  
<a href="https://github.com/Kiruthikaa2512/applied-ml-kiruthikaa/blob/main/notebooks/project04/ml04_kiruthikaa.ipynb" target="_blank">View Full Notebook</a>  

## Overview

This project explores two regression tasks:

1. **Titanic Fare Prediction** — estimating ticket prices based on passenger demographics and travel features.
2. **California Housing Price Prediction** — modeling median house values using geographic and socioeconomic indicators.

The focus is on comparing linear, regularized, and polynomial regression models, evaluating feature impact, and visualizing model performance. The California dataset provides a richer feature set, while Titanic highlights the challenges of sparse inputs.

## Datasets

### Titanic Dataset
- **Source:** `seaborn.load_dataset("titanic")`  
- **Samples:** 891  
- **Target:** `fare` (continuous)  
- **Key Features Used:**  
  - `age`, `family_size`, `sex_encoded`  
  - Engineered: `family_size = sibsp + parch + 1`

**Handling Missing Values**
- `age` filled with median  
- `embarked` filled with "missing" and one-hot encoded

### California Housing Dataset
- **Source:** `sklearn.datasets.fetch_california_housing()`  
- **Samples:** 20,640  
- **Target:** `MedHouseVal` (median house value in $100,000s)  
- **Features:** `MedInc`, `AveRooms`, `AveOccup`, `Latitude`, `Longitude`, etc.


## Project Structure

| Section | Description |
|--------|-------------|
| **1. Data Preparation** | Clean missing values, engineer features, encode categorical variables |
| **2. Feature Selection** | Compare four cases: `age`, `family_size`, `age + family_size`, `sex_encoded` |
| **3. Linear Regression** | Train and evaluate across all cases; inspect coefficients and metrics |
| **4. Model Comparison** | Ridge, ElasticNet, Polynomial Regression (degree 3 and 8) |
| **5. Visual Diagnostics** | Predicted vs actual plots, residuals, R² bar charts |
| **6. California Housing** | Train linear regression on full feature set; visualize performance |
| **7. Final Reflections** | Discuss model behavior, feature impact, and next steps |

## Titanic Regression Results (Test Set)

| Case | Features Used | R² | RMSE | MAE | Notes |
|------|----------------|-----|------|------|-------|
| 1 | age | 0.003 | 37.97 | 25.29 | Weak predictor |
| 2 | family_size | 0.022 | 37.61 | 25.03 | Slight improvement |
| 3 | age + family_size | 0.050 | 37.08 | 24.28 | Best combined case |
| 4 | sex_encoded | **0.099** | 36.10 | 24.24 | Best single feature |

## Alternative Models (Case 3)

| Model | R² | RMSE | MAE |
|-------|-----|------|------|
| Linear Regression | 0.050 | 37.08 | 24.28 |
| Ridge Regression | 0.050 | 37.08 | 24.28 |
| ElasticNet | 0.054 | 36.99 | 24.21 |
| Polynomial (deg 3) | **0.064** | **36.79** | **23.14** |

**Key Observations**
- Polynomial regression captured non-linear trends better than linear or regularized models.
- Sex alone was surprisingly predictive, likely due to class and cabin correlations.
- All models underfit due to limited features — adding `pclass`, `embarked`, or `title` could improve performance.

## California Housing Results

- **R²:** 0.608  
- **RMSE:** 0.719 (~$71,900)  
- **MAE:** 0.527 (~$52,700)  

**Visuals Included**
- Target distribution histogram  
- Predicted vs actual scatterplot  
- Residuals vs predicted plot

**Conclusion:**  
The California model performed significantly better due to richer features and larger sample size. Income and location were key drivers of housing prices.

---

## System Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Kiruthikaa2512/applied-ml-kiruthikaa.git
cd applied-ml-kiruthikaa/notebooks/project04
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

```bash
jupyter notebook ml04_kiruthikaa.ipynb
```

### 5. Convert to HTML for Hosting

```bash
jupyter nbconvert --to html ml04_kiruthikaa.ipynb --output-dir ../../docs --output ml04_kiruthikaa
```

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
```

## Key Learnings

- Regression models require careful feature selection to avoid underfitting.
- Simple features like `sex` can be surprisingly predictive.
- Polynomial regression improves fit but risks overfitting at extremes.
- Visual diagnostics are essential for interpreting model behavior.

## Future Enhancements

- Add `pclass`, `embarked`, and interaction terms to Titanic model
- Try ensemble methods (Random Forest, Gradient Boosting)
- Apply log transformation or scaling to reduce skew
- Use cross-validation for more robust evaluation
