import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

feature_names = [
    'Age at enrollment',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Scholarship holder'
]

with open("../model/random_forest_recovery_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

rf_importance = rf_model.feature_importances_

rf_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8,5))
plt.barh(rf_df['Feature'], rf_df['Importance'], color='steelblue', edgecolor='black')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


with open("../model/logistic_recovery_model.pkl", "rb") as f:
    log_model, scaler = pickle.load(f)

log_importance = log_model.coef_[0]

log_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': log_importance
}).sort_values(by='Importance', ascending=True)

colors = ['red' if val < 0 else 'green' for val in log_df['Importance']]

plt.figure(figsize=(8,5))
plt.barh(log_df['Feature'], log_df['Importance'], color=colors, edgecolor='black')
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()


with open("../model/xgboost_recovery_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

xgb_importance = xgb_model.feature_importances_

xgb_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8,5))
plt.barh(xgb_df['Feature'], xgb_df['Importance'], color='purple', edgecolor='black')
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
