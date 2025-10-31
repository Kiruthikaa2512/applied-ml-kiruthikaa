# Project 01: California Housing Price Prediction

**Author**: Kiruthikaa Natarajan Srinivasan  
**Date**: October 15, 2025  

```markdown
# Project 01 — California Housing Price Prediction

Welcome to the **California Housing ML Project** documentation.  
This project demonstrates a workflow for **data exploration**, **feature selection**, and **linear regression modeling** using the **California Housing dataset** from `scikit-learn`.

---

## Project Overview

### Objective
Predict median home values across California districts based on income and housing features.

### Dataset Summary
| Feature | Description |
|----------|--------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| **Target:** `MedHouseVal` | Median house value (in $100,000s) |

## Key Steps

1. **Import and Inspect the Data**
   - Load dataset using `fetch_california_housing`.
   - Examine data structure and summary statistics.

2. **Explore the Data**
   - Visualize distributions with histograms.
   - Use optional boxenplots and pairplots for in-depth relationships.

3. **Feature Selection**
   - Focused on key predictors: `MedInc` and `AveRooms`.

4. **Model Training**
   - Linear Regression using Scikit-Learn.
   - Evaluate model with `R²`, `MAE`, and `RMSE`.

5. **Findings**
   - Median income has the strongest correlation with housing prices.
   - Achieved **R² ≈ 0.46**, showing moderate explanatory power.

##  System Setup and Usage

### 1. Clone and Navigate
```bash
git clone https://github.com/<your-username>/ca-housing-ml.git
cd ca-housing-ml
