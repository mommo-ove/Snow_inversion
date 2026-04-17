import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 1. 读数据 & 清洗
df = pd.read_csv('Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv')
df_clean = df[df['Snow_Depth_m'] > 0].dropna()

# 2. 选定特征 (X) 和 目标 (y)
features = ['TB_18V', 'TB_18H', 'TB_23V', 'TB_23H', 'TB_36V', 'TB_36H', 'TB_89V', 'TB_89H']
X = df_clean[features]
y = df_clean['Snow_Depth_m']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. 预测与评估
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"--- 机器学习初探 ---")
print(f"随机森林预测 R^2: {r2:.3f}")
print(f"均方根误差 RMSE: {rmse:.3f} m")

# 5. 画图：预测值 vs 真实值
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, color='purple', s=10)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # 1:1 对角线
plt.title(f'Random Forest: Predicted vs Actual\n$R^2$ = {r2:.2f}, RMSE = {rmse:.3f}m')
plt.xlabel('Actual Snow Depth (m)')
plt.ylabel('Predicted Snow Depth (m)')

# 画图：特征重要性
plt.subplot(1, 2, 2)
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title('Feature Importances in Snow Depth Retrieval')
plt.xlabel('Relative Importance')

plt.tight_layout()
plt.savefig('ML_Preview.png')
print("图片已保存为 ML_Preview.png")