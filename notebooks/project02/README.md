# Project 02 — Titanic Survival Prediction

**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** October 29, 2025  

---

## Project Overview

This project explores the **Titanic dataset** to understand which factors most strongly influenced passenger survival.  
It demonstrates a full **data analysis and feature engineering workflow** — from inspection to preparation for modeling.

The project follows a clear, structured process to ensure data quality, interpretability, and balanced evaluation through stratified sampling.


## Files

- `ml02_titanic.ipynb` — Main Jupyter notebook containing the complete analysis.
- `README.md` — Repository-level summary.
- `docs/index.md` — Documentation for web or MkDocs hosting.
- (Optional) `requirements.txt` — Python dependencies.

---

## 📊 Dataset Information

- **Source:** `seaborn.load_dataset("titanic")`
- **Samples:** 891 passengers
- **Features:**
  - Numeric: `age`, `fare`, `pclass`, `sibsp`, `parch`
  - Categorical: `sex`, `embarked`, `deck`, `class`, `embark_town`
  - Target: `survived` (1 = survived, 0 = did not survive)

---

## Key Steps in the Project

1. **Import and Inspect the Data**
   - Load the Titanic dataset using Seaborn.
   - Inspect using `.info()`, `.describe()`, `.isnull().sum()`.
   - Identify missing values in `age`, `deck`, and `embark_town`.

2. **Data Exploration and Preparation**
   - Visualize distributions with scatter plots, histograms, and count plots.
   - Examine patterns in survival across class, gender, and age.
   - Handle missing data:
     - Fill `age` with the median.
     - Fill `embark_town` with the mode.

3. **Feature Engineering**
   - Create new features:
     - `family_size = sibsp + parch + 1`
     - `alone` (binary flag for passengers traveling alone)
   - Encode categorical variables (`sex`, `embarked`) numerically.

4. **Feature Selection and Target Definition**
   - Selected features: `age`, `fare`, `pclass`, `sex`, `family_size`
   - Target variable: `survived`

5. **Data Splitting**
   - Perform both **basic** and **stratified** train/test splits.
   - Compare class distributions to validate stratification effectivenes

## Findings

| Method | Non-Survivors | Survivors |
|--------|---------------|------------|
| Original | 61.6% | 38.4% |
| Basic Train | 61.1% | 38.9% |
| Basic Test | 63.7% | 36.3% |
| Stratified Train | 61.7% | 38.3% |
| Stratified Test | 61.5% | 38.5% |

- **Stratified sampling** preserves the target distribution, improving model fairness.
- **Gender, class, and age** are key predictors of survival probability.

---

How to Run

Clone the repository

git clone https://github.com/<kiruthikaa2512>/titanic-ml.git
cd titanic-ml


**Create and activate a virtual environment**

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

**Install dependencies** using pyproject.toml

Open the Notebook

``` jupyter notebook ml02_titanic.ipynb```

Run all cells sequentially to execute the analysis, feature engineering, and visualizations.

**Learning Outcomes**

Identified influential factors for survival.

Applied systematic data cleaning and feature engineering.

Demonstrated proper data splitting using stratified sampling.

Built a reproducible and interpretable preprocessing workflow.