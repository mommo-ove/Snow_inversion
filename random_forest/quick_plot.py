import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv')

# 2. 核心分析：选择一个对雪深敏感的频段 (比如 36H 或 89H)
# 物理背景：高频微波对雪层散射敏感
target_tb = 'TB_36H' 

plt.figure(figsize=(10, 6))

# 3. 绘制散点图 + 拟合线
sns.regplot(x=df[target_tb], y=df['Snow_Depth_m'], 
            scatter_kws={'alpha':0.3, 's':10, 'color':'blue'},
            line_kws={'color':'red'})

# 计算相关系数
corr = df[target_tb].corr(df['Snow_Depth_m'])

plt.title(f'IMB Data Analysis: {target_tb} vs Snow Depth\nCorrelation R = {corr:.2f}')
plt.xlabel(f'Brightness Temperature {target_tb} (K)')
plt.ylabel('Measured Snow Depth (m)')
plt.grid(True, linestyle='--', alpha=0.6)

# 4. 保存并导出
plt.savefig('imb_correlation_result.png')
print(f"分析完成！相关系数 R 为: {corr:.2f}。图片已保存。")