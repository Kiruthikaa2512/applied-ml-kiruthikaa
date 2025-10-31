# Project 02 — Kiruthikaa's Titanic Survival Prrediction + Breast Cancer Classification

**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** October 29, 2025  
**Notebook**
You can view the full Jupyter notebook for this project here:
<a href="https://github.com/Kiruthikaa251/applied-ml-kiruthikaa/blob/main/project02/project02.ipynb" target="_blank">View Full Notebook</a>

<a href="project02/project02.html" target="_blank">View Hosted Notebook</a>

---

## Overview

This project investigates two classic machine learning classification problems:

1. **Titanic Survival Prediction** — analyzing demographic and ticket-related features to predict passenger survival.
2. **Breast Cancer Classification** — using medical diagnostic data to classify tumors as malignant or benign.

Both datasets are used to demonstrate systematic data exploration, cleaning, feature engineering, model training, and evaluation — with a focus on **reproducibility**, **interpretability**, and **balanced sampling**.
---

## Datasets

### Titanic Dataset
- **Source:** `seaborn.load_dataset("titanic")`  
- **Samples:** 891  
- **Type:** Mixed (numeric + categorical)  
- **Target:** `survived` (0 = No, 1 = Yes)  
- **Key Features:**
  - `pclass`, `sex`, `age`, `fare`, `sibsp`, `parch`, `embarked`

**Handling Missing Values**
- `age`, `deck`, and `embark_town` contain missing entries.
- Missing values are imputed using the median (for numeric) and mode (for categorical).

### Breast Cancer Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569  
- **Features:** 30 numeric diagnostic attributes  
- **Target:** Binary (0 = malignant, 1 = benign)

**Goal:** Classify tumors based on measurements such as mean radius, texture, smoothness, and symmetry.

---

## Project Structure

| Section | Description |
|----------|--------------|
| **1. Import and Inspect the Data** | Load Titanic and Breast Cancer datasets; inspect types, shapes, and missing values. |
| **2. Data Exploration and Preparation** | Visualize distributions and correlations; handle nulls and outliers. |
| **3. Feature Engineering** | Create new features (e.g., `family_size`, `alone`), encode categorical data. |
| **4. Train/Test Split and Model Training** | Split using both basic and stratified sampling; train baseline classifiers. |
| **5. Bonus: Breast Cancer Classification** | Apply Logistic Regression and Decision Tree models; compare metrics. |
| **6. Final Interpretation** | Reflect on results, feature importance, and lessons learned. |

---

## Titanic Dataset Summary

| Split Type | Non-Survivors | Survivors |
|-------------|----------------|------------|
| Original | 61.6% | 38.4% |
| Stratified Train | 61.7% | 38.3% |
| Stratified Test | 61.5% | 38.5% |

- **Stratified sampling** ensures balanced class proportions.
- **Key predictors:** `sex`, `pclass`, `age`, and `fare`.
- **Insights:**
  - Female passengers had higher survival rates.
  - Higher-class passengers (1st > 2nd > 3rd) had better survival outcomes.
  - Younger passengers and those with small families fared better.
## Bonus:
## Breast Cancer Model Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | ~96% | 0.97 | 0.96 | 0.96 |
| Decision Tree | ~93% | 0.94 | 0.93 | 0.93 |

**Key Observations**
- Logistic Regression provides smoother decision boundaries and better generalization.
- Decision Tree offers interpretability through feature importance visualization.
- Important features include `mean radius`, `worst area`, and `mean concave points`.


## System Setup and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ml-projects.git
cd ml-projects/project02
````

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

```bash
jupyter notebook ml02_titanic.ipynb
jupyter notebook ml02_breast_cancer.ipynb
```

### 5. (Optional) Convert to HTML for Hosting

```bash
jupyter nbconvert --to html --TemplateExporter.exclude_input=True ml02_titanic.ipynb
jupyter nbconvert --to html --TemplateExporter.exclude_input=True ml02_breast_cancer.ipynb
```

---

## Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
```

**Environment Management:**
The project uses `venv` (or `uv`) for isolation and `nbconvert` for clean HTML exports.

---

## Key Learnings

* **Data balance** is critical for fair model evaluation.
* **Feature engineering** (e.g., family size, categorical encoding) improves model interpretability.
* **Logistic Regression** often outperforms tree-based methods on small, clean datasets.

---

## Future Enhancements

* Add **hyperparameter tuning** (GridSearchCV / RandomizedSearchCV).
* Introduce **cross-validation** for more robust performance estimates.
* Apply **feature scaling** and **PCA** for dimensionality reduction.
* Extend to ensemble methods (Random Forest, Gradient Boosting).

---

## Hosting Strategy

The notebooks can be hosted as HTML documentation using **MkDocs** or **GitHub Pages**.

Example workflow:

```bash
jupyter nbconvert --to html --TemplateExporter.exclude_input=True docs/project02/ml02_kiruthikaa.ipynb
```

Integrate the generated files in your MkDocs structure:

```yaml
nav:
  - Home: index.md
  - Project 01: project01/index.md
  - Project 02: project02/index.md
```

---

## Conclusion

Project 02 combines **data exploration** (Titanic) and **classification modeling** (Breast Cancer) to illustrate a cohesive end-to-end ML workflow.
It demonstrates **clean data handling**, **feature construction**, and **balanced evaluation** — key foundations for reliable machine learning analysis.

<!-- Rebuild after unpublishing -->

```
