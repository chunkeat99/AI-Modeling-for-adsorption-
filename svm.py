import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. LOAD DATA
# Ensure your CSV columns are named: Temperature, Time, Yield, Adsorption
try:
    df = pd.read_csv('rsm data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'rsm data.csv' not found. Please upload the file.")

# 2. SEPARATE FEATURES AND TARGETS
X = df[['temperature', 'time']]
y = df[['yield', 'adsorption']]

# 3. DATA SPLITTING
# random_state ensures you get the same results every time you run it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. FEATURE SCALING (Crucial for SVR performance)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 5. MODEL TRAINING (Support Vector Regressor)
# SVR is often better than Random Forest for very small experimental datasets
base_svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model = MultiOutputRegressor(base_svr)
model.fit(X_train_scaled, y_train)

# 6. PREDICTIONS
y_pred = model.predict(X_test_scaled)

# 7. EVALUATION AND PLOTTING
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
targets = ['Yield', 'Adsorption']

for i, col in enumerate(targets):
    # Calculate Metrics
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    
    print(f"--- {col} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
    
    # Plotting Predicted vs Actual
    ax[i].scatter(y_test.iloc[:, i], y_pred[:, i], color='blue', edgecolors='k')
    ax[i].plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 
               [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--', lw=2)
    ax[i].set_title(f'{col}: Actual vs Predicted')
    ax[i].set_xlabel('Actual Experimental Value')
    ax[i].set_ylabel('AI Predicted Value')
    ax[i].grid(True)

plt.tight_layout()
plt.show()

# 8. PREDICTION TOOL
def predict_biochar(temp, time):
    input_data = scaler_X.transform([[temp, time]])
    prediction = model.predict(input_data)
    print(f"Prediction for {temp}C and {time}hr:")
    print(f"Predicted Yield: {prediction[0][0]:.2f}%")
    print(f"Predicted Adsorption: {prediction[0][1]:.2f}%")

# Example usage:
predict_biochar(550, 1.5)