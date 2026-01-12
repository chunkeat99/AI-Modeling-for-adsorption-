import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
df = pd.read_csv('rsm data.csv')
X = df[['temperature', 'time']].values
y_yield = df['yield'].values
y_ads = df['adsorption'].values

# 2. Use 'Leave-One-Out' Cross-Validation 
# This is the gold standard for small datasets (n < 20)
def train_and_validate(X, y, name):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    
    # Gradient Boosting is often better for non-linear chemical yields
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test[0])
    
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} Optimized Results ---")
    print(f"R2 Score: {r2:.4f}")
    
    # Feature Importance
    model.fit(X, y) # Fit on all data to get importance
    importance = model.feature_importances_
    print(f"Importance - Temp: {importance[0]:.2f}, Time: {importance[1]:.2f}\n")
    return y_true, y_pred

# 3. Run for both responses
true_y, pred_y = train_and_validate(X, y_yield, "Yield")
true_a, pred_a = train_and_validate(X, y_ads, "Adsorption")

# 4. Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(true_y, pred_y, color='green')
plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], 'r--')
plt.title('Yield: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(true_a, pred_a, color='blue')
plt.plot([min(true_a), max(true_a)], [min(true_a), max(true_a)], 'r--')
plt.title('Adsorption: Actual vs Predicted')
plt.show()