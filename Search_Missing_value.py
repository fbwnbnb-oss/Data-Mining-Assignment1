import pandas as pd

# --- 数据加载 ---
print("--- 正在加载数据... ---")
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
file_path = r'C:\Users\72867\Desktop\ltjl\Census Income Data Set'

# 正常加载训练和测试数据
train_df = pd.read_csv(f'{file_path}\\adult.data', header=None, names=column_names, sep=r'\s*,\s*', engine='python',
                       na_values='?')
test_df = pd.read_csv(f'{file_path}\\adult.test', header=None, names=column_names, sep=r'\s*,\s*', engine='python',
                      skiprows=1, na_values='?')

# 合并数据集以便进行统一检查
df = pd.concat([train_df, test_df], ignore_index=True)
print("数据加载完成！\n")


# --- 核心步骤：检查每一列的缺失值数量 ---
print("--- 检查所有列的缺失值数量 ---")
missing_values_count = df.isnull().sum()

# 打印出所有包含缺失值的列（即缺失值数量大于0的列）
print("发现以下列中存在缺失值:")
print(missing_values_count[missing_values_count > 0])
