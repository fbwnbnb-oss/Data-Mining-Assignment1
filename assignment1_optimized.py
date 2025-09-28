import os
# 为了精确控制CPU核心的设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 数据加载与预处理
print("开始加载和预处理数据")
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
file_path = r'C:\Users\72867\Desktop\ltjl\Census Income Data Set'
try:
    train_df = pd.read_csv(f'{file_path}\\adult.data', header=None, names=column_names, sep=r'\s*,\s*', engine='python', na_values='?')
    test_df = pd.read_csv(f'{file_path}\\adult.test', header=None, names=column_names, sep=r'\s*,\s*', engine='python', skiprows=1, na_values='?')
except FileNotFoundError:
    print(f"错误：无法在路径 '{file_path}' 中找到数据文件。请检查路径是否正确。")
    exit()
test_df['income'] = test_df['income'].str.replace('.', '', regex=False)
df = pd.concat([train_df, test_df], ignore_index=True)
for col in ['workclass', 'occupation', 'native-country']:
    df[col].fillna(df[col].mode()[0], inplace=True)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df.drop('income', axis=1)
y = df['income']
num_train_samples = len(train_df)
X_train, X_test = X[:num_train_samples], X[num_train_samples:]
y_train, y_test = y[:num_train_samples], y[num_train_samples:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("数据准备完成！\n")

# 决策树调优
print("开始进行超参数调优")
print("正在调优决策树...")
param_grid_dt = {
    'max_depth': [5, 10, 15, None], 'min_samples_leaf': [1, 5, 10], 'criterion': ['gini', 'entropy']
}
# 使用 n_jobs=-1 cpu全核心运行
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, n_jobs=-1, verbose=1)
grid_search_dt.fit(X_train_scaled, y_train)
best_dt = grid_search_dt.best_estimator_
print(f"决策树最佳参数: {grid_search_dt.best_params_}")

# SVM 调优
print("\n始优化SVM调优过程")

# 创建一个数据子集用于快速调优
# 取20%的数据来进行搜索，加快速度（若不取，训练时长大幅增加）
# stratify=y_train 确保抽样后，收入>50K和<=50K的比例和原始数据一致
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train_scaled, y_train, train_size=0.2, random_state=42, stratify=y_train
)
print(f"创建了一个大小为 {len(y_train_sample)} 的样本用于SVM快速调优...")

# 在数据子集上进行快速随机搜索
print("\n正在调优SVM... (使用数据子集加速)")
param_dist_svm = {
    'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']
}
# 降低 n_iter 和 cv 进一步加速
random_search_svm = RandomizedSearchCV(SVC(random_state=42, probability=True), param_dist_svm, n_iter=5, cv=3,
                                       n_jobs=-1, verbose=1, random_state=42)
# 使用 sample 数据集进行 fit
random_search_svm.fit(X_train_sample, y_train_sample)
best_params_svm = random_search_svm.best_params_
print(f"在样本上找到的SVM最佳参数: {best_params_svm}\n")

# 使用找到的最佳参数，在完整的训练集上训练最终的SVM模型
print("使用最佳参数在【完整训练集】上训练最终的SVM模型...")
start_train_time = time.time()
# 使用上面找到的 best_params_svm 来创建最终模型
final_svm = SVC(random_state=42, probability=True, **best_params_svm)
final_svm.fit(X_train_scaled, y_train)
end_train_time = time.time()
print(f"最终SVM模型训练完成，耗时: {end_train_time - start_train_time:.2f} 秒")


# 最终模型进行评估
print("\n使用调优后的最佳模型进行评估")
models = {
    "Tuned Decision Tree": best_dt,
    "Tuned SVM (Optimized)": final_svm # 使用我们最终训练好的SVM模型
}
results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"正在评估模型: {name}")
    start_pred_time = time.time()
    y_pred = model.predict(X_test_scaled)
    end_pred_time = time.time()
    pred_time = end_pred_time - start_pred_time
    report = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
    print(f"\n{name} 分类报告:")
    print(report)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Prediction Time (s)': pred_time,
        'AUC': roc_auc
    }
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

print("\n评估完成")

# 结果可视化与汇总
print("\n最终结果")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (假正率)')
plt.ylabel('True Positive Rate (真正率)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('roc_curves.png')
print("ROC 曲线图已保存为 'roc_curves.png'")
results_df = pd.DataFrame(results).T
print("\n性能对比")
print(results_df)
