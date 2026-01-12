import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
#plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备
data = pd.DataFrame({
    'temperature': [400, 800, 400, 800, 400, 800, 600, 600, 600, 600, 600, 600],
    'time': [1, 1, 3, 3, 2, 2, 1, 3, 2, 2, 2, 2],
    'yield': [23.21, 14.96, 20.28, 5.69, 19.67, 12.87, 20.21, 14.59, 15.93, 16.63, 19.11, 18.76],
    'adsorption': [94.075, 98.26, 95.385, 98.575, 94.955, 99.405, 98.09, 99.235, 98.885, 98.58, 99.15, 99.12]
})

print("=" * 60)
print("数据概览")
print("=" * 60)
print(data.describe())
print("\n数据形状:", data.shape)

# 2. 数据可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 相关性热图
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
axes[0, 0].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 温度 vs 产率
axes[0, 1].scatter(data['temperature'], data['yield'], c=data['time'], cmap='viridis', s=100)
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Yield')
axes[0, 1].set_title('Temperature vs Yield (colored by time)')
axes[0, 1].grid(True, alpha=0.3)

# 时间 vs 吸附
axes[1, 0].scatter(data['time'], data['adsorption'], c=data['temperature'], cmap='plasma', s=100)
axes[1, 0].set_xlabel('Time (h)')
axes[1, 0].set_ylabel('Adsorption')
axes[1, 0].set_title('Time vs Adsorption (colored by temperature)')
axes[1, 0].grid(True, alpha=0.3)

# 产率 vs 吸附
axes[1, 1].scatter(data['yield'], data['adsorption'], c=data['temperature'], cmap='coolwarm', s=100)
axes[1, 1].set_xlabel('Yield')
axes[1, 1].set_ylabel('Adsorption')
axes[1, 1].set_title('Yield vs Adsorption (colored by temperature)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 准备特征和目标变量
# 情况1: 预测 yield
X_yield = data[['temperature', 'time']].values
y_yield = data['yield'].values

# 情况2: 预测 adsorption
X_adsorption = data[['temperature', 'time']].values
y_adsorption = data['adsorption'].values

# 数据标准化
scaler_X = StandardScaler()
scaler_y_yield = StandardScaler()
scaler_y_adsorption = StandardScaler()

X_scaled = scaler_X.fit_transform(X_yield)
y_yield_scaled = scaler_y_yield.fit_transform(y_yield.reshape(-1, 1)).ravel()
y_adsorption_scaled = scaler_y_adsorption.fit_transform(y_adsorption.reshape(-1, 1)).ravel()

# 4. 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.1)
}

# 5. 使用交叉验证评估模型（针对小数据集）
def evaluate_models(X, y, target_name):
    print("\n" + "=" * 60)
    print(f"预测目标: {target_name}")
    print("=" * 60)
    
    results = {}
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 3折交叉验证（数据较少）
    
    for name, model in models.items():
        # 交叉验证评分
        cv_scores = cross_val_score(model, X, y, cv=kfold, 
                                     scoring='neg_mean_squared_error')
        cv_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        
        rmse = np.sqrt(-cv_scores.mean())
        r2 = cv_r2.mean()
        
        results[name] = {'RMSE': rmse, 'R2': r2}
        
        print(f"\n{name}:")
        print(f"  交叉验证 RMSE: {rmse:.4f} (±{np.sqrt(-cv_scores).std():.4f})")
        print(f"  交叉验证 R²: {r2:.4f} (±{cv_r2.std():.4f})")
    
    return results

# 评估两个目标变量
results_yield = evaluate_models(X_scaled, y_yield_scaled, "Yield (产率)")
results_adsorption = evaluate_models(X_scaled, y_adsorption_scaled, "Adsorption (吸附)")

# 6. 可视化模型性能比较
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Yield 模型比较
model_names = list(results_yield.keys())
rmse_yield = [results_yield[m]['RMSE'] for m in model_names]
r2_yield = [results_yield[m]['R2'] for m in model_names]

x_pos = np.arange(len(model_names))
axes[0].bar(x_pos, rmse_yield, color='steelblue', alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].set_ylabel('RMSE (Scaled)')
axes[0].set_title('Model Performance: Yield Prediction')
axes[0].grid(True, alpha=0.3, axis='y')

# Adsorption 模型比较
rmse_adsorption = [results_adsorption[m]['RMSE'] for m in model_names]
r2_adsorption = [results_adsorption[m]['R2'] for m in model_names]

axes[1].bar(x_pos, rmse_adsorption, color='coral', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].set_ylabel('RMSE (Scaled)')
axes[1].set_title('Model Performance: Adsorption Prediction')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 训练最佳模型并进行预测示例
print("\n" + "=" * 60)
print("使用最佳模型进行预测示例")
print("=" * 60)

# 选择表现最好的模型（Random Forest）
best_model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
best_model_adsorption = RandomForestRegressor(n_estimators=100, random_state=42)

best_model_yield.fit(X_scaled, y_yield_scaled)
best_model_adsorption.fit(X_scaled, y_adsorption_scaled)

# 新数据预测示例
new_data = np.array([[500, 1.5], [700, 2.5]])  # 温度500°C/时间1.5h 和 700°C/2.5h
new_data_scaled = scaler_X.transform(new_data)

pred_yield_scaled = best_model_yield.predict(new_data_scaled)
pred_adsorption_scaled = best_model_adsorption.predict(new_data_scaled)

pred_yield = scaler_y_yield.inverse_transform(pred_yield_scaled.reshape(-1, 1)).ravel()
pred_adsorption = scaler_y_adsorption.inverse_transform(pred_adsorption_scaled.reshape(-1, 1)).ravel()

print("\n新条件预测:")
for i, (temp, time) in enumerate(new_data):
    print(f"\n温度={temp}°C, 时间={time}h:")
    print(f"  预测产率 (Yield): {pred_yield[i]:.2f}")
    print(f"  预测吸附 (Adsorption): {pred_adsorption[i]:.2f}")

# 8. 特征重要性（针对 Random Forest）
print("\n" + "=" * 60)
print("特征重要性分析 (Random Forest)")
print("=" * 60)

feature_importance_yield = best_model_yield.feature_importances_
feature_importance_adsorption = best_model_adsorption.feature_importances_
features = ['Temperature', 'Time']

print("\nYield 预测的特征重要性:")
for feat, imp in zip(features, feature_importance_yield):
    print(f"  {feat}: {imp:.4f}")

print("\nAdsorption 预测的特征重要性:")
for feat, imp in zip(features, feature_importance_adsorption):
    print(f"  {feat}: {imp:.4f}")

# 9. 特征重要性可视化
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

# 10. 实际值 vs 预测值对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Yield 预测
y_pred_yield_all = best_model_yield.predict(X_scaled)
y_pred_yield_original = scaler_y_yield.inverse_transform(y_pred_yield_all.reshape(-1, 1)).ravel()

axes[0].scatter(y_yield, y_pred_yield_original, s=100, alpha=0.7, color='steelblue', edgecolors='black')
axes[0].plot([y_yield.min(), y_yield.max()], [y_yield.min(), y_yield.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Yield')
axes[0].set_ylabel('Predicted Yield')
axes[0].set_title('Yield: Actual vs Predicted')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Adsorption 预测
y_pred_adsorption_all = best_model_adsorption.predict(X_scaled)
y_pred_adsorption_original = scaler_y_adsorption.inverse_transform(y_pred_adsorption_all.reshape(-1, 1)).ravel()

axes[1].scatter(y_adsorption, y_pred_adsorption_original, s=100, alpha=0.7, color='coral', edgecolors='black')
axes[1].plot([y_adsorption.min(), y_adsorption.max()], [y_adsorption.min(), y_adsorption.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Adsorption')
axes[1].set_ylabel('Predicted Adsorption')
axes[1].set_title('Adsorption: Actual vs Predicted')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 残差分析图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Yield 残差
residuals_yield = y_yield - y_pred_yield_original

# 残差 vs 预测值
axes[0, 0].scatter(y_pred_yield_original, residuals_yield, s=100, alpha=0.7, color='steelblue', edgecolors='black')
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Yield')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Yield: Residual Plot')
axes[0, 0].grid(True, alpha=0.3)

# 残差分布直方图
axes[0, 1].hist(residuals_yield, bins=5, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Yield: Residual Distribution')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Adsorption 残差
residuals_adsorption = y_adsorption - y_pred_adsorption_original

# 残差 vs 预测值
axes[1, 0].scatter(y_pred_adsorption_original, residuals_adsorption, s=100, alpha=0.7, color='coral', edgecolors='black')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Adsorption')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Adsorption: Residual Plot')
axes[1, 0].grid(True, alpha=0.3)

# 残差分布直方图
axes[1, 1].hist(residuals_adsorption, bins=5, color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Adsorption: Residual Distribution')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 3D 响应面图（预测不同温度和时间组合的结果）
from mpl_toolkits.mplot3d import Axes3D

# 创建网格数据
temp_range = np.linspace(400, 800, 30)
time_range = np.linspace(1, 3, 30)
temp_grid, time_grid = np.meshgrid(temp_range, time_range)

# 准备预测数据
grid_data = np.c_[temp_grid.ravel(), time_grid.ravel()]
grid_data_scaled = scaler_X.transform(grid_data)

# 预测
yield_pred_grid = best_model_yield.predict(grid_data_scaled)
yield_pred_grid = scaler_y_yield.inverse_transform(yield_pred_grid.reshape(-1, 1)).ravel()
yield_pred_grid = yield_pred_grid.reshape(temp_grid.shape)

adsorption_pred_grid = best_model_adsorption.predict(grid_data_scaled)
adsorption_pred_grid = scaler_y_adsorption.inverse_transform(adsorption_pred_grid.reshape(-1, 1)).ravel()
adsorption_pred_grid = adsorption_pred_grid.reshape(temp_grid.shape)

# 绘制3D图
fig = plt.figure(figsize=(16, 6))

# Yield 3D 响应面
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(temp_grid, time_grid, yield_pred_grid, cmap='viridis', alpha=0.8)
ax1.scatter(data['temperature'], data['time'], data['yield'], c='red', s=100, edgecolors='black', label='Actual Data')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Time (h)')
ax1.set_zlabel('Yield')
ax1.set_title('3D Response Surface: Yield Prediction')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Adsorption 3D 响应面
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

# 13. 等高线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Yield 等高线
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