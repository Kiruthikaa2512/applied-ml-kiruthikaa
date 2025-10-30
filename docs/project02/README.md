# Project 02: Titanic + Breast Cancer Classification

**Author**: Kiruthikaa Natarajan Srinivasan  
**Date**: October 29, 2025  

## Overview

This project explores survival prediction using the Titanic dataset and includes a bonus section analyzing the Breast Cancer dataset. The goal is to identify meaningful predictors, apply classification models, and evaluate performance using stratified sampling and clear reflections.

## Datasets

- **Titanic**: Loaded from the `seaborn` library for consistency. Includes age, fare, class, gender, and survival status.
- **Breast Cancer**: Loaded from `sklearn.datasets`. Includes numeric features and binary classification target.

## Project Structure

The notebook follows a structured format with clearly numbered sections:

1. Import and inspect the data  
2. Data exploration and preparation  
3. Feature selection and justification  
4. Train/test split and model training  
5. Bonus: Breast Cancer classification and comparison  
6. Final interpretation and closure

Each section includes markdown reflections aligned with rubric questions.

## Requirements

This project uses the following Python libraries:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Environment setup and reproducibility were managed using `uv`, `venv`, and `nbconvert`.

## Hosting Strategy

- Notebook was converted to HTML using:

  ```bash
  jupyter nbconvert --to html --TemplateExporter.exclude_input=True docs/project02/ml02_kiruthikaa.ipynb