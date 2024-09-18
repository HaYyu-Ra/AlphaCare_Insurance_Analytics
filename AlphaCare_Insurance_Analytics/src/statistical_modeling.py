# statistical_modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/insurance_data.csv')

# Data Preparation
# Handling Missing Data
data = data.fillna(data.median())  # Impute missing values with median for numerical columns
data = data.dropna()  # Drop rows with missing values in any column

# Feature Engineering
# Create new features (example: total premium, total claims)
data['TotalPremium'] = data['revenue']
data['TotalClaims'] = data['claims_paid']

# Encoding Categorical Data
data_encoded = pd.get_dummies(data, columns=['province', 'zip_code', 'gender'], drop_first=True)

# Define features and target variable
X = data_encoded.drop(['TotalPremium', 'TotalClaims'], axis=1)
y_premium = data_encoded['TotalPremium']
y_claims = data_encoded['TotalClaims']

# Train-Test Split
X_train, X_test, y_train_premium, y_test_premium = train_test_split(X, y_premium, test_size=0.3, random_state=42)
X_train, X_test, y_train_claims, y_test_claims = train_test_split(X, y_claims, test_size=0.3, random_state=42)

# Model Building
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_premium)
y_pred_premium_lin = lin_reg.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_premium)
y_pred_premium_rf = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train_premium)
y_pred_premium_xgb = xgb.predict(X_test)

# Model Evaluation
print("Linear Regression:")
print(f"Mean Squared Error: {mean_squared_error(y_test_premium, y_pred_premium_lin)}")
print(f"R^2 Score: {r2_score(y_test_premium, y_pred_premium_lin)}\n")

print("Random Forest:")
print(f"Mean Squared Error: {mean_squared_error(y_test_premium, y_pred_premium_rf)}")
print(f"R^2 Score: {r2_score(y_test_premium, y_pred_premium_rf)}\n")

print("XGBoost:")
print(f"Mean Squared Error: {mean_squared_error(y_test_premium, y_pred_premium_xgb)}")
print(f"R^2 Score: {r2_score(y_test_premium, y_pred_premium_xgb)}\n")

# Feature Importance Analysis
print("Feature Importance - Random Forest:")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]}")

# SHAP Analysis
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

# Plot SHAP values
shap.summary_plot(shap_values, X_test)

# Save results for further analysis
pd.DataFrame({'Actual': y_test_premium, 'Predicted_LR': y_pred_premium_lin, 'Predicted_RF': y_pred_premium_rf, 'Predicted_XGB': y_pred_premium_xgb}).to_csv('results/predictions.csv', index=False)

print("Statistical Modeling and Evaluation complete. Results have been saved to 'results/predictions.csv'.")
