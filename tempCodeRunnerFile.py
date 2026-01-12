import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 1. LOAD DATA
df = pd.read_csv('rsm data.csv')

# 2. CHOOSE RESPONSE
# You can change this to 'Yield' or 'Adsorption'
target_col = 'yield' 

X = df[['temperature', 'time']].values
y = df[target_col].values

# 3. DEFINE MODELS
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42),
    "SVM (SVR)": SVR(kernel='rbf', C=10, epsilon=0.01)
}

# 4. CROSS-VALIDATION LOOP (LOO)
results = {}
predictions_map = {}

for name, model in models.items():
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scaling is mandatory for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred.append(model.predict(X_test_scaled)[0])
        y_true.append(y_test[0])
    
    results[name] = r2_score(y_true, y_pred)
    predictions_map[name] = y_pred

# 5. VISUALIZATION
plt.figure(figsize=(15, 5))
colors = ['#FF9999', '#66B2FF', '#99FF99']

for i, (name, r2) in enumerate(results.items()):
    plt.subplot(1, 3, i+1)
    plt.scatter(y, predictions_map[name], color=colors[i], edgecolors='k', s=100)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Fit')
    plt.title(f"{name}\n$R^2$ = {r2:.3f}")
    plt.xlabel('Experimental (Actual)')
    plt.ylabel('AI Predicted')
    plt.legend()

plt.tight_layout()
plt.show()

print("Comparison Summary:")
for name, r2 in results.items():
    print(f"{name} R2: {r2:.4f}")