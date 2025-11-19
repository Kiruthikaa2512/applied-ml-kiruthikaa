# Project 05 — Kiruthikaa's Ensemble Models on Wine Quality  
**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** November 18, 2025  

**Project Overview:**
This project focuses on the **UCI Red Wine Quality dataset** and uses **ensemble machine learning models** to classify wine quality into three groups: **low**, **medium**, and **high**. The goal is to understand how ensemble methods behave on a real-world dataset containing physicochemical measurements of wine.

The notebook walks through the complete modeling workflow, including data preprocessing, feature preparation, model training, performance evaluation, visual diagnostics, and model comparison. Accuracy, weighted F1 score, and confusion matrices are used to assess performance.

Random Forest and Gradient Boosting are the two main ensemble models evaluated in this project.

## Files

* `ensemble-wine-kiruthikaa.ipynb` — Main notebook for the classification workflow
* `README.md` — Summary of the project
* `docs/index.md` — Documentation for hosting or portfolio use
* (Optional) `requirements.txt` — Python dependencies if publishing or sharing

## Dataset Information

### Wine Quality Dataset

* **Source:** UCI Machine Learning Repository
* **Samples:** 1599 red wine records
* **Features:** 11 numeric physicochemical properties
  (acidities, residual sugar, sulphates, alcohol, etc.)
* **Target:** `quality` (originally 0–10, converted into 3 categories)

### Target Engineering

To make the predictions more meaningful, the original scores are grouped into:

* **Low quality:** 3–4
* **Medium quality:** 5–6
* **High quality:** 7–8

These labels are also mapped to numeric categories (0, 1, 2) for modeling.

## Key Steps in the Project

### 1. Data Preparation

* Loaded and inspected the raw dataset
* Converted the original integer quality scores into label categories
* Created a numeric target column for classification
* Dropped non-essential columns during feature selection
* Explored distributions of labels, numeric classes, and original quality scores
* Visualized distributions using histograms, boxplots, and count plots
* Confirmed class imbalance (majority in medium category)

### 2. Train-Test Split

* Used an 80/20 split
* Applied stratification to preserve class proportions
* Ensured consistent reproducibility with `random_state=42`

### 3. Modeling

Two ensemble models were trained and compared:

#### **Random Forest (100 trees)**

* Strong baseline ensemble model
* Produces feature importance values
* Very high training accuracy and strong test accuracy
* Shows some overfitting based on performance gaps

#### **Gradient Boosting (100 estimators)**

* More controlled learning through boosting
* Slightly lower accuracy than Random Forest
* Smaller gap between training and testing metrics
* Error curve plotted across boosting stages

### 4. Visual Diagnostics

* Confusion matrix heatmaps for both models
* Feature importance plot from Random Forest
* Error vs number of trees plot for Gradient Boosting
* Comparisons using a consolidated results table
* Added gap calculations between training and testing scores for both accuracy and F1

### 5. Findings

* **Random Forest** achieved the highest overall accuracy and F1 on the test set.
* **Gradient Boosting** showed more stable generalization with fewer signs of overfitting.
* Both models struggled with the low-quality class due to its small sample size.
* Alcohol, volatile acidity, and sulphates appeared as the most influential features.

## How to Run

Clone the repository:

```bash
git clone https://github.com/<kiruthikaa2512>/wine-ensemble-classification.git
cd wine-ensemble-classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebook:

```bash
jupyter notebook ensemble-wine-kiruthikaa.ipynb
```

Run all cells in order to reproduce the classification workflow.

## Learning Outcomes

* Practiced converting a continuous rating scale into classification categories
* Built and compared ensemble models on a real-world prediction task
* Evaluated both accuracy and generalization using confusion matrices and F1 scores
* Gained experience interpreting model behavior through visualizations
* Learned how ensemble methods handle imbalance and numeric-only features
* Strengthened skills in communicating results and comparing modeling strategies
