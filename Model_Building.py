# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Load data

import pandas as pd

df = pd.read_csv("Telco.csv")

# Convert churn to binary

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop useless column(s)

df.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges to numeric, drop rows with 0 value

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Bin tenure, drop unbinned column

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0,12,24,48,72],
    labels=["0-12","12-24","24-48","48+"]
)
df.drop (columns=["tenure"], inplace=True)

# One-hot encoding, drop one column to avoid multicollinearity
df_encoded = pd.get_dummies(df, drop_first=True)

# Split data

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Random forest

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 10]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

rf_probs = best_rf.predict_proba(X_test)[:,1]

# Linear regression

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)

param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    log_model,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid.fit(X_train, y_train)

best_log = grid.best_estimator_

# evaluate using test data

probs_rf = best_rf.predict_proba(X_test)[:, 1]
probs_log = best_log.predict_proba(X_test)[:, 1]

preds_rf = best_rf.predict(X_test)
preds_log = best_log.predict(X_test)

roc_auc_rf = roc_auc_score(y_test, probs_rf)
roc_auc_log = roc_auc_score(y_test, probs_log)

roc_auc_data = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "ROC_AUC": [roc_auc_rf, roc_auc_log]
})

roc_auc_data.to_csv("roc_auc_scores.csv", index=False)

# confusion matrices

cm_rf = pd.DataFrame(
    confusion_matrix(y_test, preds_rf),
    columns=["Predicted_No", "Predicted_Yes"],
    index=["Actual_No", "Actual_Yes"]
)
cm_rf.to_csv("cm_rf.csv")

cm_log = pd.DataFrame(
    confusion_matrix(y_test, preds_log),
    columns=["Predicted_No", "Predicted_Yes"],
    index=["Actual_No", "Actual_Yes"]
)
cm_log.to_csv("cm_log.csv")

# print evaluations

print("Random Forest ROC-AUC:", roc_auc_score(y_test, probs_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, preds_rf))
print(classification_report(y_test, preds_rf))

print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, probs_log))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, preds_log))
print(classification_report(y_test, preds_log))

# plot and save curves

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, probs_rf)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.savefig("roc_curve_rf.png", dpi=300)
plt.show()

roc_data_rf = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr
})

roc_data_rf.to_csv("roc_curve_data_rf.csv", index=False)

fpr, tpr, _ = roc_curve(y_test, probs_log)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.savefig("roc_curve_log.png", dpi=300)
plt.show()

roc_data_log = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr
})

roc_data_log.to_csv("roc_curve_data_log.csv", index=False)

# save models

import joblib

joblib.dump(best_rf, "random_forest_churn_model.pkl")
joblib.dump(best_log, "logistic_regression_churn_model.pkl")

# save training/testing data

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)