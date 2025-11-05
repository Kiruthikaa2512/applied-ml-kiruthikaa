# Project 03: Titanic Survival Prediction — Classifier Comparison

**Author:** Kiruthikaa Natarajan Srinivasan  
**Date:** November 4, 2025

## Overview

This project explores the Titanic dataset to predict passenger survival using three classification models: Decision Tree, Support Vector Machine (SVM), and Neural Network (MLP). The goal is to evaluate how different models and feature combinations affect classification performance. The project also includes a bonus experiment using the Breast Cancer Wisconsin dataset to test model generalization.

## Objectives

- Implement and evaluate three classification models:
  - Decision Tree Classifier
  - Support Vector Machine (SVM) with multiple kernels
  - Multi-layer Perceptron (MLP) Neural Network
- Compare model performance across three feature sets:
  - Case 1: `alone`
  - Case 2: `age`
  - Case 3: `age + family_size`
- Visualize decision boundaries and support vectors
- Reflect on model behavior, feature impact, and generalization

## Dataset

The Titanic dataset was loaded from seaborn and includes:
- `survived`: Target variable (0 = did not survive, 1 = survived)
- `age`: Passenger age
- `sibsp`, `parch`: Number of siblings/spouses and parents/children aboard
- `alone`: Boolean indicating if the passenger was traveling alone
- `family_size`: Engineered feature combining `sibsp` and `parch` plus one

Missing values in `age` and `embark_town` were imputed using median and mode, respectively. Categorical variables were mapped to numeric values.

## Feature Selection

Three feature sets were defined:

- **Case 1:** `alone` — captures social isolation
- **Case 2:** `age` — reflects prioritization in evacuation
- **Case 3:** `age + family_size` — combines demographic and relational context

These features were chosen for their interpretability and relevance to survival scenarios.

## Model Performance Summary (Test Set)

| Model Type             | Case   | Features Used        | Accuracy | Precision | Recall | F1-Score | Notes |
|------------------------|--------|-----------------------|----------|-----------|--------|----------|-------|
| Decision Tree          | Case 1 | alone                | 63.00%   | 64.00%    | 63.00% | 63.00%   | Balanced baseline with a single feature |
|                        | Case 2 | age                  | 61.00%   | 58.00%    | 61.00% | 55.00%   | Weak recall for survivors |
|                        | Case 3 | age + family_size    | 59.00%   | 57.00%    | 59.00% | 57.00%   | Overfitting observed |
| SVM (RBF Kernel)       | Case 1 | alone                | 63.00%   | 64.00%    | 63.00% | 63.00%   | Balanced performance |
|                        | Case 2 | age                  | 63.00%   | 66.00%    | 63.00% | 52.00%   | High precision, poor recall |
|                        | Case 3 | age + family_size    | 63.00%   | 66.00%    | 63.00% | 52.00%   | No gain from extra feature |
| SVM (Linear Kernel)    | Case 3 | age + family_size    | 61.00%   | 38.00%    | 61.00% | 47.00%   | Predicted only non-survivors |
| SVM (Poly Kernel)      | Case 3 | age + family_size    | 61.00%   | 38.00%    | 61.00% | 47.00%   | Similar to linear kernel |
| SVM (Sigmoid Kernel)   | Case 3 | age + family_size    | 55.00%   | 55.00%    | 55.00% | 55.00%   | More balanced, lower accuracy |
| Neural Network (MLP)   | Case 3 | age + family_size    | 66.00%   | 65.00%    | 66.00% | 65.00%   | Best overall performance |

## Visualizations

- **Decision Trees**: Plotted and saved for all three cases
  - `tree_case1_alone.png`
  - `tree_case2_age.png`
  - `tree_case3_age_family.png`
- **SVM Support Vectors**: Visualized for all three cases
- **Neural Network Decision Surface**: Contour plot showing predicted survival zones overlaid with test data

## Reflections

### Model Behavior

- SVC models had high precision for non-survivors but low recall for survivors.
- The sigmoid kernel was more balanced but less accurate.
- The neural network offered the best balance and smoothest decision boundary.

### Surprises

- Adding `family_size` to `age` did not improve performance.
- The neural network did not outperform SVCs despite its complexity.

### Why Some Models Performed Better

- SVCs are sensitive to kernel choice and class imbalance.
- Neural networks need more data or tuning to reach full potential.
- Simpler models like `alone`-based Decision Trees performed surprisingly well.

## Final Insights

| Model              | Setup                         | Accuracy | Key Insight                                      |
|--------------------|-------------------------------|----------|--------------------------------------------------|
| SVC (RBF kernel)   | Default settings              | 63%      | Good at predicting non-survivors, missed survivors |
| SVC (Linear)       | Default settings              | 61%      | Struggled with survivor predictions              |
| SVC (Polynomial)   | Degree 3                      | 61%      | Similar to linear, minimal gain                  |
| SVC (Sigmoid)      | Default settings              | 55%      | More balanced, but lower accuracy                |
| Neural Network     | (50,25,10), solver='lbfgs'    | 63%      | Smooth decision boundary, slightly better balance |

## Challenges

- **Imbalanced Data**: Most models favored the majority class.
- **Limited Features**: Only three features limited model expressiveness.
- **Model Sensitivity**: SVCs were highly sensitive to kernel and parameter choices.

## Next Steps

- **Hyperparameter Tuning**: Adjust depth, C/gamma, layer sizes, learning rates
- **Feature Expansion**: Add `pclass`, `sex`, `fare`
- **Try Ensemble Models**: Random Forest, Gradient Boosting, Voting Classifiers
- **Improve Evaluation**: Use cross-validation for more reliable performance estimates

## Bonus Experiment: Breast Cancer Dataset

To test generalization, the same workflow was applied to the Breast Cancer Wisconsin dataset (569 samples, 30 numeric features, binary classification).

### Performance Summary

| Model              | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
|--------------------|----------|------------------|---------------|----------------|
| Decision Tree      | 91%      | 90%              | 92%           | 91%            |
| SVC (RBF Kernel)   | 93%      | 93%              | 91%           | 93%            |
| Neural Network     | 94%      | 94%              | 93%           | 94%            |

### Interpretation

- All models performed well, confirming the dataset’s suitability for classification.
- SVC and MLP outperformed the Decision Tree, which showed signs of overfitting.
- Feature scaling and tuning could further enhance performance.

## Author

**Kiruthikaa Natarajan Srinivasan**  
GitHub: [Kiruthikaa2512](https://github.com/Kiruthikaa2512)