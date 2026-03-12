#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:53:32 2026

@author: tobyh1
"""
import pandas as pd

# load models and data

import joblib

model_rf = joblib.load("random_forest_churn_model.pkl")
model_log = joblib.load("logistic_regression_churn_model.pkl")

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# determine most important features: random forest

import pandas as pd

importances = model_rf.feature_importances_

train_data = X_train.copy()
train_data["Churn"] = y_train.values

means={}
for mean in X_train.columns:
    means[mean]=train_data.groupby(mean)["Churn"].mean().iloc[-1]

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
})

feature_importance["mean"] = feature_importance["feature"].map(means)

churnyn = {}

baseline = train_data["Churn"].mean()

for i in means:
    if means[i] >= baseline:
        churnyn[i] = "Churn"
    else:
        churnyn[i] = "Retain"

feature_importance["churn_yes_or_no"] = feature_importance["feature"].map(churnyn)

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance = feature_importance.round(3)

feature_importance = feature_importance.reset_index(drop=True)

print(feature_importance)

print("Average churn rate: ", train_data["Churn"].mean())

feature_importance.to_csv("feature_importance.csv", index=False)