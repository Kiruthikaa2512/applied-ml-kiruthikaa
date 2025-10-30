
# Lab 2: Howell Dataset Exploration and Preparation

## Purpose
This lab focuses on exploring and preparing the Howell dataset for future modeling tasks. It introduces foundational data analysis techniques including inspection, visualization, feature engineering, and train/test splitting. The goal is to understand the structure and patterns in the data while building a reproducible workflow.

## Project Structure
```
notebooks/
├── Lab2_Howell.ipynb       # Main analysis notebook
├── Howell.csv              # Dataset file (semicolon-separated)
└── README.md               # Project overview and instructions
```

## Dataset Overview
- **Source**: Howell.csv
- **Separator**: Semicolon (`;`)
- **Features**: `height`, `weight`, `age`, `male`
- **Target (for future modeling)**: `male` (binary classification)

## Key Steps
1. **Data Import and Inspection**
   - Load dataset with `pandas`
   - Check structure, missing values, and summary statistics
   - Analyze feature correlations

2. **Data Exploration**
   - Use scatter matrix to visualize distributions and relationships
   - Create scatter plots (height vs weight, age vs height)
   - Color plots by gender for deeper insights

3. **Data Cleaning**
   - Demonstrate handling of missing values (even if none exist)
   - Compute and impute median/mean values

4. **Feature Engineering**
   - Add BMI feature using metric formula
   - Categorize BMI into `Underweight`, `Normal`, `Overweight`, `Obese`
   - Visualize BMI trends by age and gender

5. **Data Splitting**
   - Split dataset into adults and children (age > 18)
   - Perform basic and stratified train/test splits
   - Compare gender ratios across splits to evaluate fairness

## 📈 Visualizations
- Scatter matrix of `height`, `weight`, `age`
- Scatter plots colored by gender
- Masked plots for adult height vs weight
- Age vs BMI plot

## Reflections
- Observations on distributions, correlations, and anomalies
- Importance of class balance in train/test splits
- When and why to use stratified sampling

## Next Steps
This lab sets the foundation for modeling tasks in future labs, where classification models will be trained to predict gender based on physical attributes.

