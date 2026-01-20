import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
data = pd.DataFrame({
    'temperature': [400, 800, 400, 800, 400, 800, 600, 600, 600, 600, 600, 600],
    'time': [1, 1, 3, 3, 2, 2, 1, 3, 2, 2, 2, 2],
    'yield': [23.21, 14.96, 20.28, 5.69, 19.67, 12.87, 20.21, 14.59, 15.93, 16.63, 19.11, 18.76],
    'adsorption': [94.075, 98.26, 95.385, 98.575, 94.955, 99.405, 98.09, 99.235, 98.885, 98.58, 99.15, 99.12]
})

print("=" * 80)
print("DATA OVERVIEW")
print("=" * 80)
print(data.describe())
print("\nData shape:", data.shape)
print("\nData info:")
print(data.info())

# ============================================================================
# 2. DATA VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
axes[0, 0].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# Temperature vs Yield
axes[0, 1].scatter(data['temperature'], data['yield'], c=data['time'], cmap='viridis', s=100)
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Yield')
axes[0, 1].set_title('Temperature vs Yield (colored by time)')
axes[0, 1].grid(True, alpha=0.3)

# Time vs Adsorption
axes[1, 0].scatter(data['time'], data['adsorption'], c=data['temperature'], cmap='plasma', s=100)
axes[1, 0].set_xlabel('Time (h)')
axes[1, 0].set_ylabel('Adsorption')
axes[1, 0].set_title('Time vs Adsorption (colored by temperature)')
axes[1, 0].grid(True, alpha=0.3)

# Yield vs Adsorption
axes[1, 1].scatter(data['yield'], data['adsorption'], c=data['temperature'], cmap='coolwarm', s=100)
axes[1, 1].set_xlabel('Yield')
axes[1, 1].set_ylabel('Adsorption')
axes[1, 1].set_title('Yield vs Adsorption (colored by temperature)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. PREPARE FEATURES AND TARGET VARIABLES
# ============================================================================
# Case 1: Predict yield
X_yield = data[['temperature', 'time']].values
y_yield = data['yield'].values

# Case 2: Predict adsorption
X_adsorption = data[['temperature', 'time']].values
y_adsorption = data['adsorption'].values

# Data standardization (important for SVR and neural networks)
scaler_X = StandardScaler()
scaler_y_yield = StandardScaler()
scaler_y_adsorption = StandardScaler()

X_scaled = scaler_X.fit_transform(X_yield)
y_yield_scaled = scaler_y_yield.fit_transform(y_yield.reshape(-1, 1)).ravel()
y_adsorption_scaled = scaler_y_adsorption.fit_transform(y_adsorption.reshape(-1, 1)).ravel()

# ============================================================================
# 4. DEFINE MACHINE LEARNING MODELS
# ============================================================================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.1)
}

# ============================================================================
# 5. COMPREHENSIVE ACCURACY METRICS FUNCTION
# ============================================================================
def calculate_accuracy_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive accuracy metrics for regression models
    
    Metrics included:
    - RMSE: Root Mean Squared Error (lower is better)
    - MAE: Mean Absolute Error (lower is better)
    - MAPE: Mean Absolute Percentage Error (lower is better)
    - R²: Coefficient of Determination (higher is better, max=1)
    - Adjusted R²: Adjusted for number of predictors
    - Max Error: Maximum residual error
    """
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf
    
    # Max error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Adjusted R² (assuming 2 features)
    n = len(y_true)
    p = 2  # number of predictors
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    # Mean residual (should be close to 0)
    mean_residual = np.mean(y_true - y_pred)
    
    # Standard deviation of residuals
    std_residual = np.std(y_true - y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'Max Error': max_error,
        'Mean Residual': mean_residual,
        'Std Residual': std_residual
    }
    
    return metrics

def print_accuracy_report(metrics, model_name, target_name):
    """Print formatted accuracy report"""
    print(f"\n{'='*70}")
    print(f"{model_name} - {target_name} Prediction")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Value':<15} {'Interpretation'}")
    print(f"{'-'*70}")
    print(f"{'RMSE':<25} {metrics['RMSE']:<15.4f} Lower is better")
    print(f"{'MAE':<25} {metrics['MAE']:<15.4f} Lower is better")
    print(f"{'MAPE (%)':<25} {metrics['MAPE (%)']:<15.2f} Lower is better")
    print(f"{'R²':<25} {metrics['R²']:<15.4f} Closer to 1 is better")
    print(f"{'Adjusted R²':<25} {metrics['Adjusted R²']:<15.4f} Adjusted for features")
    print(f"{'Max Error':<25} {metrics['Max Error']:<15.4f} Largest prediction error")
    print(f"{'Mean Residual':<25} {metrics['Mean Residual']:<15.4f} Should be close to 0")
    print(f"{'Std Residual':<25} {metrics['Std Residual']:<15.4f} Consistency of errors")
    print(f"{'='*70}")

# ============================================================================
# 6. MODEL EVALUATION WITH CROSS-VALIDATION
# ============================================================================
def evaluate_models(X, y, target_name):
    """
    Evaluate all models using K-Fold cross-validation
    Returns: Dictionary of results for each model
    """
    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION RESULTS: {target_name}")
    print("=" * 80)
    
    results = {}
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold CV for small dataset
    
    for name, model in models.items():
        # Cross-validation scores
        cv_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        cv_mae = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        cv_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        
        rmse_mean = np.sqrt(cv_mse.mean())
        rmse_std = np.sqrt(cv_mse).std()
        mae_mean = cv_mae.mean()
        mae_std = cv_mae.std()
        r2_mean = cv_r2.mean()
        r2_std = cv_r2.std()
        
        results[name] = {
            'RMSE': rmse_mean,
            'RMSE_std': rmse_std,
            'MAE': mae_mean,
            'MAE_std': mae_std,
            'R2': r2_mean,
            'R2_std': r2_std
        }
        
        print(f"\n{name}:")
        print(f"  CV RMSE: {rmse_mean:.4f} (±{rmse_std:.4f})")
        print(f"  CV MAE:  {mae_mean:.4f} (±{mae_std:.4f})")
        print(f"  CV R²:   {r2_mean:.4f} (±{r2_std:.4f})")
    
    return results

# Evaluate both target variables
results_yield = evaluate_models(X_scaled, y_yield_scaled, "Yield")
results_adsorption = evaluate_models(X_scaled, y_adsorption_scaled, "Adsorption")

# ============================================================================
# 7. VISUALIZE MODEL PERFORMANCE COMPARISON
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Yield model comparison
model_names = list(results_yield.keys())
rmse_yield = [results_yield[m]['RMSE'] for m in model_names]
r2_yield = [results_yield[m]['R2'] for m in model_names]

x_pos = np.arange(len(model_names))
axes[0].bar(x_pos, rmse_yield, color='steelblue', alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].set_ylabel('RMSE (Scaled)')
axes[0].set_title('Model Performance: Yield Prediction (Lower is Better)')
axes[0].grid(True, alpha=0.3, axis='y')

# Adsorption model comparison
rmse_adsorption = [results_adsorption[m]['RMSE'] for m in model_names]
r2_adsorption = [results_adsorption[m]['R2'] for m in model_names]

axes[1].bar(x_pos, rmse_adsorption, color='coral', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].set_ylabel('RMSE (Scaled)')
axes[1].set_title('Model Performance: Adsorption Prediction (Lower is Better)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. TRAIN BEST MODELS AND DETAILED ACCURACY ASSESSMENT
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING BEST MODELS FOR DETAILED ACCURACY ASSESSMENT")
print("=" * 80)

# Select best performing model (Random Forest)
best_model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
best_model_adsorption = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models on full dataset
best_model_yield.fit(X_scaled, y_yield_scaled)
best_model_adsorption.fit(X_scaled, y_adsorption_scaled)

# Make predictions on training data
y_pred_yield_scaled = best_model_yield.predict(X_scaled)
y_pred_adsorption_scaled = best_model_adsorption.predict(X_scaled)

# Inverse transform to original scale
y_pred_yield_original = scaler_y_yield.inverse_transform(y_pred_yield_scaled.reshape(-1, 1)).ravel()
y_pred_adsorption_original = scaler_y_adsorption.inverse_transform(y_pred_adsorption_scaled.reshape(-1, 1)).ravel()

# Calculate comprehensive metrics for Yield
metrics_yield = calculate_accuracy_metrics(y_yield, y_pred_yield_original, "Random Forest")
print_accuracy_report(metrics_yield, "Random Forest", "Yield")

# Calculate comprehensive metrics for Adsorption
metrics_adsorption = calculate_accuracy_metrics(y_adsorption, y_pred_adsorption_original, "Random Forest")
print_accuracy_report(metrics_adsorption, "Random Forest", "Adsorption")

# ============================================================================
# 9. DETAILED PREDICTION VS ACTUAL COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED PREDICTION VS ACTUAL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Temperature': data['temperature'],
    'Time': data['time'],
    'Actual_Yield': y_yield,
    'Predicted_Yield': y_pred_yield_original,
    'Yield_Error': y_yield - y_pred_yield_original,
    'Yield_Error_%': np.abs((y_yield - y_pred_yield_original) / y_yield * 100),
    'Actual_Adsorption': y_adsorption,
    'Predicted_Adsorption': y_pred_adsorption_original,
    'Adsorption_Error': y_adsorption - y_pred_adsorption_original,
    'Adsorption_Error_%': np.abs((y_adsorption - y_pred_adsorption_original) / y_adsorption * 100)
})

print("\nYield Predictions:")
print(comparison_df[['Temperature', 'Time', 'Actual_Yield', 'Predicted_Yield', 'Yield_Error', 'Yield_Error_%']].to_string(index=False))

print("\n\nAdsorption Predictions:")
print(comparison_df[['Temperature', 'Time', 'Actual_Adsorption', 'Predicted_Adsorption', 'Adsorption_Error', 'Adsorption_Error_%']].to_string(index=False))

# ============================================================================
# 10. ACCURACY METRICS VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Yield metrics bar chart
yield_metric_names = ['RMSE', 'MAE', 'MAPE (%)', 'Max Error']
yield_metric_values = [metrics_yield['RMSE'], metrics_yield['MAE'], 
                       metrics_yield['MAPE (%)'], metrics_yield['Max Error']]

axes[0, 0].bar(yield_metric_names, yield_metric_values, color='steelblue', alpha=0.7)
axes[0, 0].set_ylabel('Value')
axes[0, 0].set_title('Yield Prediction Error Metrics (Lower is Better)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Yield R² visualization
axes[0, 1].bar(['R²', 'Adjusted R²'], [metrics_yield['R²'], metrics_yield['Adjusted R²']], 
               color=['steelblue', 'darkblue'], alpha=0.7)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Yield Prediction R² Scores (Higher is Better, Max=1)')
axes[0, 1].set_ylim([0, 1.1])
axes[0, 1].axhline(y=1, color='r', linestyle='--', label='Perfect Score')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].legend()

# Adsorption metrics bar chart
adsorption_metric_values = [metrics_adsorption['RMSE'], metrics_adsorption['MAE'], 
                            metrics_adsorption['MAPE (%)'], metrics_adsorption['Max Error']]

axes[1, 0].bar(yield_metric_names, adsorption_metric_values, color='coral', alpha=0.7)
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Adsorption Prediction Error Metrics (Lower is Better)')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Adsorption R² visualization
axes[1, 1].bar(['R²', 'Adjusted R²'], [metrics_adsorption['R²'], metrics_adsorption['Adjusted R²']], 
               color=['coral', 'darkred'], alpha=0.7)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Adsorption Prediction R² Scores (Higher is Better, Max=1)')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Perfect Score')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('accuracy_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 11. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS (Random Forest)")
print("=" * 80)

feature_importance_yield = best_model_yield.feature_importances_
feature_importance_adsorption = best_model_adsorption.feature_importances_
features = ['Temperature', 'Time']

print("\nYield Prediction Feature Importance:")
for feat, imp in zip(features, feature_importance_yield):
    print(f"  {feat}: {imp:.4f} ({imp*100:.2f}%)")

print("\nAdsorption Prediction Feature Importance:")
for feat, imp in zip(features, feature_importance_adsorption):
    print(f"  {feat}: {imp:.4f} ({imp*100:.2f}%)")

# Feature importance visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].barh(features, feature_importance_yield, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Feature Importance: Yield Prediction')
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(features, feature_importance_adsorption, color='coral', alpha=0.7)
axes[1].set_xlabel('Importance Score')
axes[1].set_title('Feature Importance: Adsorption Prediction')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 12. ACTUAL VS PREDICTED PLOTS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Yield prediction
axes[0].scatter(y_yield, y_pred_yield_original, s=100, alpha=0.7, color='steelblue', edgecolors='black')
axes[0].plot([y_yield.min(), y_yield.max()], [y_yield.min(), y_yield.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Yield')
axes[0].set_ylabel('Predicted Yield')
axes[0].set_title(f'Yield: Actual vs Predicted (R²={metrics_yield["R²"]:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Add annotations for points far from diagonal
for i in range(len(y_yield)):
    error_pct = comparison_df.iloc[i]['Yield_Error_%']
    if error_pct > 10:  # Annotate if error > 10%
        axes[0].annotate(f'{error_pct:.1f}%', 
                        (y_yield[i], y_pred_yield_original[i]),
                        fontsize=8, alpha=0.7)

# Adsorption prediction
axes[1].scatter(y_adsorption, y_pred_adsorption_original, s=100, alpha=0.7, color='coral', edgecolors='black')
axes[1].plot([y_adsorption.min(), y_adsorption.max()], [y_adsorption.min(), y_adsorption.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Adsorption')
axes[1].set_ylabel('Predicted Adsorption')
axes[1].set_title(f'Adsorption: Actual vs Predicted (R²={metrics_adsorption["R²"]:.3f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Add annotations for points far from diagonal
for i in range(len(y_adsorption)):
    error_pct = comparison_df.iloc[i]['Adsorption_Error_%']
    if error_pct > 1:  # Annotate if error > 1%
        axes[1].annotate(f'{error_pct:.1f}%', 
                        (y_adsorption[i], y_pred_adsorption_original[i]),
                        fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 13. RESIDUAL ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Yield residuals
residuals_yield = y_yield - y_pred_yield_original

# Residuals vs Predicted values
axes[0, 0].scatter(y_pred_yield_original, residuals_yield, s=100, alpha=0.7, color='steelblue', edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Yield')
axes[0, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[0, 0].set_title('Yield: Residual Plot')
axes[0, 0].grid(True, alpha=0.3)

# Residual distribution histogram
axes[0, 1].hist(residuals_yield, bins=5, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Yield: Residual Distribution (Mean={metrics_yield["Mean Residual"]:.2f})')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Adsorption residuals
residuals_adsorption = y_adsorption - y_pred_adsorption_original

# Residuals vs Predicted values
axes[1, 0].scatter(y_pred_adsorption_original, residuals_adsorption, s=100, alpha=0.7, color='coral', edgecolors='black')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Adsorption')
axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[1, 0].set_title('Adsorption: Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# Residual distribution histogram
axes[1, 1].hist(residuals_adsorption, bins=5, color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Adsorption: Residual Distribution (Mean={metrics_adsorption["Mean Residual"]:.2f})')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 14. 3D RESPONSE SURFACE PLOTS
# ============================================================================
from mpl_toolkits.mplot3d import Axes3D

# Create grid data for prediction surface
temp_range = np.linspace(400, 800, 30)
time_range = np.linspace(1, 3, 30)
temp_grid, time_grid = np.meshgrid(temp_range, time_range)

# Prepare prediction data
grid_data = np.c_[temp_grid.ravel(), time_grid.ravel()]
grid_data_scaled = scaler_X.transform(grid_data)

# Predict on grid
yield_pred_grid = best_model_yield.predict(grid_data_scaled)
yield_pred_grid = scaler_y_yield.inverse_transform(yield_pred_grid.reshape(-1, 1)).ravel()
yield_pred_grid = yield_pred_grid.reshape(temp_grid.shape)

adsorption_pred_grid = best_model_adsorption.predict(grid_data_scaled)
adsorption_pred_grid = scaler_y_adsorption.inverse_transform(adsorption_pred_grid.reshape(-1, 1)).ravel()
adsorption_pred_grid = adsorption_pred_grid.reshape(temp_grid.shape)

# Plot 3D surfaces
fig = plt.figure(figsize=(16, 6))

# Yield 3D response surface
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(temp_grid, time_grid, yield_pred_grid, cmap='viridis', alpha=0.8)
ax1.scatter(data['temperature'], data['time'], data['yield'], c='red', s=100, edgecolors='black', label='Actual Data')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Time (h)')
ax1.set_zlabel('Yield')
ax1.set_title('3D Response Surface: Yield Prediction')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Adsorption 3D response surface
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(temp_grid, time_grid, adsorption_pred_grid, cmap='plasma', alpha=0.8)
ax2.scatter(data['temperature'], data['time'], data['adsorption'], c='red', s=100, edgecolors='black', label='Actual Data')
ax2.set_xlabel('Temperature (°C)')
ax2.set_ylabel('Time (h)')
ax2.set_zlabel('Adsorption')
ax2.set_title('3D Response Surface: Adsorption Prediction')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.savefig('3d_response_surface.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 15. CONTOUR PLOTS
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Yield contour plot
contour1 = axes[0].contourf(temp_grid, time_grid, yield_pred_grid, levels=15, cmap='viridis', alpha=0.8)
axes[0].scatter(data['temperature'], data['time'], s=100, c='red', edgecolors='black', label='Actual Data', zorder=5)
axes[0].set_xlabel('Temperature (°C)')
axes[0].set_ylabel('Time (h)')
axes[0].set_title('Contour Plot: Yield Prediction')
axes[0].legend()
fig.colorbar(contour1, ax=axes[0], label='Yield')

# Adsorption 等高线
contour2 = axes[1].contourf(temp_grid, time_grid, adsorption_pred_grid, levels=15, cmap='plasma', alpha=0.8)
axes[1].scatter(data['temperature'], data['time'], s=100, c='red', edgecolors='black', label='Actual Data', zorder=5)
axes[1].set_xlabel('Temperature (°C)')
axes[1].set_ylabel('Time (h)')
axes[1].set_title('Contour Plot: Adsorption Prediction')
axes[1].legend()
fig.colorbar(contour2, ax=axes[1], label='Adsorption')

plt.tight_layout()
plt.savefig('contour_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("分析完成！已保存所有可视化图表：")
print("  1. data_visualization.png - 数据探索图")
print("  2. model_comparison.png - 模型性能比较")
print("  3. feature_importance.png - 特征重要性")
print("  4. actual_vs_predicted.png - 实际值vs预测值")
print("  5. residual_analysis.png - 残差分析")
print("  6. 3d_response_surface.png - 3D响应面")
print("  7. contour_plots.png - 等高线图")
print("=" * 60)
