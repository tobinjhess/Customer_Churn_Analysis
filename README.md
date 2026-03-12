<img width="1041" height="607" alt="image" src="https://github.com/user-attachments/assets/4f74eef5-9fc2-47d3-9bf7-569fbb630a61" />


# Customer_Churn_Analysis
This project analyzes customer churn using classification models and exploratory data analysis to identify patterns associated with customer attrition. The goal is to understand which factors are most associated with churn and evaluate model performance using ROC curves and AUC metrics.

**Project Overview**

Customer churn is a critical metric for many businesses because retaining existing customers is often less costly than acquiring new ones. This analysis investigates behavioral and demographic factors associated with churn and compares the predictive performance of multiple classification models.

The project includes data preprocessing, exploratory data analysis, feature evaluation, and model performance comparison.

**Dataset**

The dataset contains customer-level information including demographic attributes, service usage, and account details. These features are used to analyze churn behavior and evaluate predictive models.

Typical features include:
  *Customer tenure
  *Contract type
  *Monthly charges
  *Payment method
  *Internet service type
  *Service add-ons
  *Churn status (target variable)

**Analysis Pipeline**

The project follows a typical machine learning workflow:
  *Data cleaning and preprocessing
  *Exploratory data analysis
  *Feature inspection and transformation
  *Model training and evaluation
  *Performance comparison using ROC curves and AUC

Python was used for data preparation and model evaluation.

**Models Evaluated**

Multiple classification models were evaluated to assess predictive performance and compare trade-offs between sensitivity and specificity.

Examples include:
  *Logistic Regression
  *Random Forest

Model performance was compared using ROC curves and Area Under the Curve (AUC).

**Model Evaluation**

ROC curves were used to visualize classifier performance across decision thresholds.

Key evaluation metrics include:
  *ROC Curve
  *AUC Score
  *Confusion Matrix

**Results**

The analysis highlights the near identical predictive performance of the two models and identifies features associated with higher churn risk. These insights can help organizations identify at-risk customers and design targeted retention strategies.

**Tools Used**

  *Python
  *Pandas
  *NumPy
  *Matplotlib
  *scikit-learn
