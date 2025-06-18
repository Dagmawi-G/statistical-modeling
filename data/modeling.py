# File: C:\Users\hp\Desktop\Week Three\modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap
import os

# Pull dataset from DVC
os.system('dvc pull')

# Load dataset
try:
    data = pd.read_csv('data/insurance.csv')
except FileNotFoundError:
    print("Warning: Using placeholder dataset.")
    data = pd.DataFrame({
        'PolicyID': range(1, 1001),
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], 1000),
        'PostalCode': np.random.choice(['2000', '8000', '4000'], 1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck'], 1000),
        'TotalPremium': np.random.uniform(500, 2000, 1000),
        'ClaimAmount': np.random.choice([0, np.random.uniform(100, 10000)], 1000, p=[0.8, 0.2]),
        'CalculatedPremiumPerTerm': np.random.uniform(100, 500, 1000)
    })

# Check columns and update as needed
print("Columns:", data.columns.tolist())
# Update these based on actual column names
claim_col = 'ClaimAmount'  # Replace with actual column (e.g., 'Claims', 'TotalClaims')
premium_col = 'TotalPremium'  # Replace with actual column
target_premium = 'CalculatedPremiumPerTerm'  # Replace if different

# Data Preparation
# Handle missing values
data = data.fillna(data.select_dtypes(include=np.number).mean())  # Numeric: mean
data = data.fillna(data.select_dtypes(exclude=np.number).mode().iloc[0])  # Categorical: mode

# Feature Engineering
data['LossRatio'] = data[claim_col] / data[premium_col].replace(0, 1)  # Avoid division by zero
data['ClaimOccurred'] = data[claim_col] > 0

# Encode categorical variables
categorical_cols = ['Province', 'PostalCode', 'Gender', 'VehicleType']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Select features
features = ['Province', 'PostalCode', 'Gender', 'VehicleType', 'TotalPremium', 'LossRatio']
if not all(col in data.columns for col in features):
    print("Warning: Some features missing. Using available numeric/categorical columns.")
    features = [col for col in data.columns if col not in [claim_col, target_premium, 'PolicyID', 'ClaimOccurred']]

# 1. Claim Severity Prediction (Regression for TotalClaims > 0)
claim_data = data[data['ClaimOccurred']]
if len(claim_data) > 0:
    X = claim_data[features]
    y = claim_data[claim_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'RMSE': rmse, 'R-squared': r2})
        print(f"{name} - RMSE: {rmse:.2f}, R-squared: {r2:.2f}")

    # Feature Importance (XGBoost with SHAP)
    xgb_model = models['XGBoost']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    # Save SHAP plot (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.savefig('shap_feature_importance.png')
    plt.close()

    # Save results
    with open('modeling_results.txt', 'w') as f:
        f.write("Task 4: Predictive Modeling Results\n\n")
        for r in results:
            f.write(f"Model: {r['Model']}\nRMSE: {r['RMSE']:.2f}\nR-squared: {r['R-squared']:.2f}\n\n")
        f.write("Feature Importance: See shap_feature_importance.png\n")
else:
    print("No claims data available for severity prediction.")

# 2. Premium Optimization (Regression for CalculatedPremiumPerTerm)
X = data[features]
y = data[target_premium]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
premium_model = XGBRegressor(random_state=42)
premium_model.fit(X_train, y_train)
y_pred = premium_model.fit(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Premium Prediction - RMSE: {rmse:.2f}, R-squared: {r2:.2f}")

# Save premium results
with open('modeling_results.txt', 'a') as f:
    f.write("Premium Optimization\n")
    f.write(f"XGBoost - RMSE: {rmse:.2f}, R-squared: {r2:.2f}\n")