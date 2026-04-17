import pandas as pd
df = pd.read_csv('Validation_Dataset_IMB_AllBuoys_Combined_AllData.csv')
df_clean = df[df['Snow_Depth_m'] > 0] # 排除无雪情况

# 1. 计算物理参数 GR
df_clean['GR_36_18'] = (df_clean['TB_36V'] - df_clean['TB_18V']) / (df_clean['TB_36V'] + df_clean['TB_18V'])

# 2. 提取所有微波频段 + 目标(雪深)
tb_cols = ['TB_18V', 'TB_18H', 'TB_23V', 'TB_23H', 'TB_36V', 'TB_36H', 'TB_89V', 'TB_89H', 'GR_36_18']

print("--- 各微波通道与实测雪深的相关系数 (R) ---")
for col in tb_cols:
    corr = df_clean[col].corr(df_clean['Snow_Depth_m'])
    print(f"{col}: {corr:.3f}")
    