import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
df = pd.read_csv('RSM data.csv')
X = df[['temperature', 'time']]
y_yield = df['yield']
y_ads = df['adsorption']

# We use Leave-One-Out (LOO) Cross-Validation because the dataset is small (n=13)
def train_rf(X, y, target_name):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    
    # Initialize Random Forest
    # n_estimators=50 is enough for small data to prevent overfitting
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf.fit(X_train, y_train)
        y_pred.append(rf.predict(X_test)[0])
        y_true.append(y_test.values[0])
    
    # Calculate Final Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"--- {target_name} Results ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
    
    return y_true, y_pred, rf

# Execute for both responses
true_y, pred_y, model_yield = train_rf(X, y_yield, "Yield")
true_a, pred_a, model_ads = train_rf(X, y_ads, "Adsorption")

# --- GRAPHING SECTION ---
plt.figure(figsize=(14, 10))

# Plot 1: Yield Actual vs Predicted
plt.subplot(2, 2, 1)
plt.scatter(true_y, pred_y, color='forestgreen', s=60, edgecolors='k')
plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], 'r--', lw=2)
plt.title('Yield: Actual vs Predicted')
plt.xlabel('Experimental Yield (%)')
plt.ylabel('AI Predicted Yield (%)')

# Plot 2: Adsorption Actual vs Predicted
plt.subplot(2, 2, 2)
plt.scatter(true_a, pred_a, color='royalblue', s=60, edgecolors='k')
plt.plot([min(true_a), max(true_a)], [min(true_a), max(true_a)], 'r--', lw=2)
plt.title('Adsorption: Actual vs Predicted')
plt.xlabel('Experimental Adsorption (%)')
plt.ylabel('AI Predicted Adsorption (%)')

# Plot 3: Feature Importance (Which factor matters more?)
importances = model_ads.feature_importances_
plt.subplot(2, 2, 3)
sns.barplot(x=['Temperature', 'Time'], y=importances, palette='viridis')
plt.title('Feature Importance (Adsorption)')
plt.ylabel('Weight')

plt.tight_layout()
plt.show()