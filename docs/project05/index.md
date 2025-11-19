# Wine Quality Classification with Ensemble Models

**Author:** Kiruthikaa Natarajan Srinivasan
**Date:** November 18, 2025

**Notebook**  
You can view the full Jupyter notebook for this project here:  
<a href="https://github.com/Kiruthikaa2512/applied-ml-kiruthikaa/blob/main/notebooks/project05/ensemble-kiruthikaa.ipynb" target="_blank">View Full Notebook</a>  

## Overview

This project applies ensemble machine learning methods to classify red wine quality using data from the UCI Wine Quality dataset. The task is to predict whether a wine falls into the low, medium, or high quality category based on its physicochemical characteristics.

The notebook follows a complete workflow from exploring the dataset to preparing features, training models, evaluating results, and comparing model performance. The two main ensemble models used are Random Forest and Gradient Boosting.


## 1. Understanding the Dataset

The dataset includes 1599 samples of red wine and 11 numeric features such as acidity, sulphates, and alcohol content.
The original `quality` score ranges from 0 to 10, but the differences between adjacent scores are small, so I converted the values into three broader categories:

* **Low quality:** 3–4
* **Medium quality:** 5–6
* **High quality:** 7–8

These were also mapped to numeric labels (0, 1, 2) for model training.

I visualized the distribution of the labels, examined histograms of the quality scores, and used boxplots to show the relationship between each label and the original numeric quality.

## 2. Preparing the Data

The data preparation steps included:

* Creating the categorical and numeric target variables
* Selecting all 11 chemical attributes as features
* Dropping the original quality column
* Splitting the data into training and testing sets with an 80/20 ratio
* Using **stratified sampling** to maintain the class proportions

Stratification was particularly important because the dataset is imbalanced, with the medium-quality class dominating.

## 3. Choosing the Models

Two ensemble methods were chosen for evaluation:

### Random Forest (100 Trees)

This model was used as a strong baseline. It works well with numerical features and provides feature importance, which helps with interpretation.

### Gradient Boosting (100 Estimators)

This model builds trees sequentially and tends to handle difficult cases more effectively. It often offers better generalization compared to bagging.

Both models were trained using the same helper function to ensure consistency in evaluation.

## 4. Training and Evaluation

Each model was evaluated on:

* Training accuracy
* Test accuracy
* Weighted F1 score
* Confusion matrices

These results helped show not only how well each model performed but also how well it generalized.

Additional diagnostics included:

* **Feature importance plot** for Random Forest
* **Error vs number of trees** plot for Gradient Boosting
* **Confusion matrix heatmaps** to visualize misclassifications

These visualizations made it easier to interpret each model’s strengths and weaknesses.

## 5. Comparing Performance

I created a consolidated results table containing accuracy, F1 scores, and gap values (difference between training and test metrics). Sorting the table by test accuracy provided a clear view of which model performed best.

Key observations:

* Random Forest achieved the highest accuracy on the test set.
* Gradient Boosting was slightly lower in accuracy but had smaller gaps, showing more stable generalization.
* Both models had difficulty predicting the low-quality class because of its small sample size.
* Alcohol, volatile acidity, and sulphates stood out as important features.

## 6. Key Insights

A few main takeaways from the project:

* Bagging models like Random Forest can deliver high accuracy but may overfit when trees become highly specialized.
* Boosting models build on errors and often generalize better on imbalanced data.
* Class imbalance significantly affects performance, especially on minority classes.
* High test accuracy does not always mean all classes are being predicted well.
* Feature importance helps highlight which chemical properties contribute most to wine quality predictions.

## 7. Next Steps

If I continue improving the model, I would explore:

* Oversampling methods such as SMOTE
* Hyperparameter tuning for both ensemble models
* Trying advanced boosting algorithms such as XGBoost or LightGBM
* Experimenting with probability calibration
* Adding multiclass ROC curves for deeper evaluation

These steps could help improve performance, particularly on the minority classes.

## Conclusion

This project helped me practice applying ensemble models to a real classification problem. Working through the full pipeline—data exploration, feature engineering, model building, evaluation, and comparison—gave me a clearer understanding of how ensemble methods behave and how to interpret their results. The project also emphasized the importance of communicating findings clearly and basing conclusions on data. 
