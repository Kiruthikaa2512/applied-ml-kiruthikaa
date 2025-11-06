# Project 03 — Kiruthikaa’s Titanic Classifier + Breast Cancer Bonus

**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** November 4, 2025  

**Notebook**  
You can view the full Jupyter notebook for this project here:  
<a href="https://github.com/Kiruthikaa2512/applied-ml-kiruthikaa/blob/main/notebooks/project03/ml03_kiruthikaa.ipynb" target="_blank">View Full Notebook</a>  

---

## Overview

This project explores two classification tasks:

1. **Titanic Survival Prediction** — using social and demographic features to predict passenger survival.
2. **Breast Cancer Classification** — applying machine learning to medical diagnostic data to classify tumors.

The focus is on comparing multiple models (Decision Tree, SVM, Neural Network), evaluating feature impact, and visualizing decision boundaries. The bonus experiment tests model generalization on a well-separated dataset.

---

## Datasets

### Titanic Dataset
- **Source:** `seaborn.load_dataset("titanic")`  
- **Samples:** 891  
- **Target:** `survived` (0 = No, 1 = Yes)  
- **Key Features Used:**  
  - `alone`, `age`, `family_size` (engineered from `sibsp` + `parch`)

**Handling Missing Values**
- `age` and `embark_town` were imputed using median and mode respectively.

### Breast Cancer Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`  
- **Samples:** 569  
- **Features:** 30 numeric diagnostic attributes  
- **Target:** Binary (0 = malignant, 1 = benign)

---

## Project Structure

| Section | Description |
|--------|-------------|
| **1. Data Preparation** | Clean missing values, engineer `family_size`, encode categorical features |
| **2. Feature Selection** | Compare three cases: `alone`, `age`, `age + family_size` |
| **3. Decision Tree Models** | Train and evaluate across all cases; visualize trees and confusion matrices |
| **4. SVM Models** | Compare kernels (RBF, linear, poly, sigmoid); visualize support vectors |
| **5. Neural Network** | Train MLP on Case 3; visualize decision surface |
| **6. Bonus: Breast Cancer Classification** | Apply all models to medical dataset; compare metrics |
| **7. Final Reflections** | Discuss model behavior, feature impact, and next steps |

---

## Titanic Model Comparison (Test Set)

| Model Type | Case | Features Used | Accuracy | Precision | Recall | F1-Score | Notes |
|------------|------|----------------|----------|-----------|--------|----------|-------|
| Decision Tree | 1 | alone | 63% | 64% | 63% | 63% | Balanced baseline |
| Decision Tree | 2 | age | 61% | 58% | 61% | 55% | Weak recall for survivors |
| Decision Tree | 3 | age + family_size | 59% | 57% | 59% | 57% | Overfitting observed |
| SVC (RBF) | 1 | alone | 63% | 64% | 63% | 63% | Balanced |
| SVC (RBF) | 2 | age | 63% | 66% | 63% | 52% | High precision, poor recall |
| SVC (RBF) | 3 | age + family_size | 63% | 66% | 63% | 52% | No gain from extra feature |
| SVC (Linear) | 3 | age + family_size | 61% | 38% | 61% | 47% | Predicted only non-survivors |
| SVC (Poly) | 3 | age + family_size | 61% | 38% | 61% | 47% | Similar to linear |
| SVC (Sigmoid) | 3 | age + family_size | 55% | 55% | 55% | 55% | More balanced, lower accuracy |
| Neural Network | 3 | age + family_size | 66% | 65% | 66% | 65% | Best overall performance |

---

## Breast Cancer Model Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 91% | 90% | 92% | 91% |
| SVC (RBF) | 93% | 93% | 91% | 93% |
| Neural Network | 94% | 94% | 93% | 94% |

**Key Observations**
- All models performed well due to clear class separation.
- Neural Network had the best balance and generalization.
- Feature scaling could further improve SVC and MLP performance.

---

## System Setup and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Kiruthikaa2512/applied-ml-kiruthikaa.git
cd applied-ml-kiruthikaa/notebooks/project03
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
jupyter notebook ml03_kiruthikaa.ipynb
```

### 5. Convert to HTML for Hosting

```bash
jupyter nbconvert --to html --TemplateExporter.exclude_input=True ml03_kiruthikaa.ipynb
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

---

## Key Learnings

- Simple features like `alone` can outperform more complex combinations.
- Class imbalance affects precision and recall — especially in SVM.
- Neural networks offer smoother decision boundaries but require tuning.
- Visualizations reveal model behavior beyond raw metrics.

---

## Future Enhancements

- Add hyperparameter tuning (GridSearchCV)
- Use cross-validation for reliability
- Include more features (`pclass`, `sex`, `fare`)
- Try ensemble methods (Random Forest, Gradient Boosting)


