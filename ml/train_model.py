import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import m2cgen as m2c
from xgboost import XGBClassifier
import csv

malware_csv = 'data/malware_features.csv'
whitelist_csv = 'data/whitelist_features.csv'

# 手动读取CSV文件并自动填充缺失字段
def read_csv_with_padding(file_path):
    print(f"开始读取 {file_path}...")
    max_cols = 0
    rows = []
    
    # 首先确定最大列数
    with open(file_path, 'r', encoding='latin1', errors='replace') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            max_cols = max(max_cols, len(row))
            rows.append(row)
    
    print(f"文件 {file_path} 最大列数: {max_cols}")
    
    # 为每一行填充缺失的字段
    padded_rows = []
    for row in rows:
        # 如果行长度小于最大列数，用'0'填充
        padded_row = row + ['0'] * (max_cols - len(row))
        padded_rows.append(padded_row)
    
    # 转换为DataFrame
    df = pd.DataFrame(padded_rows)
    print(f"读取 {file_path} 完成，形状: {df.shape}")
    return df

# 读取CSV文件
malware_data = read_csv_with_padding(malware_csv)
whitelist_data = read_csv_with_padding(whitelist_csv)

# 删除第一列（路径列）
malware_data = malware_data.iloc[:, 1:]
whitelist_data = whitelist_data.iloc[:, 1:]

# 将所有列转换为数值类型，非数值将转为NaN
for col in malware_data.columns:
    malware_data[col] = pd.to_numeric(malware_data[col], errors='coerce')
for col in whitelist_data.columns:
    whitelist_data[col] = pd.to_numeric(whitelist_data[col], errors='coerce')

# 用0填充NaN值
malware_data.fillna(0, inplace=True)
whitelist_data.fillna(0, inplace=True)

# 找到最大列数（最长的特征向量）
max_cols = max(malware_data.shape[1], whitelist_data.shape[1])

# 用 0 填充（Padding）数据，使所有样本的列数相同
malware_data = malware_data.reindex(columns=range(max_cols), fill_value=0)
whitelist_data = whitelist_data.reindex(columns=range(max_cols), fill_value=0)

# 添加标签
malware_data['label'] = 1  # 恶意软件
whitelist_data['label'] = 0  # 白名单（正常）
print(malware_data.head())
print(whitelist_data.head())

# 合并数据
combined_data = pd.concat([malware_data, whitelist_data], ignore_index=True)
print(f"合并后数据形状: {combined_data.shape}")

# 分离特征和标签
X = combined_data.drop('label', axis=1)
y = combined_data['label']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

# 创建 XGBoost 数据集
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 训练 XGBoost 模型
num_rounds = 30
# 创建watchlist来监控训练和验证集的性能
watchlist = [(dtrain, '训练集'), (dtest, '验证集')]
pos_ratio = np.mean(y_train)  # 计算 1 的比例

clf = XGBClassifier(
    base_score=pos_ratio,  #

    objective='binary:logistic',  # 适用于二分类
    max_depth=6,  # 树的最大深度
    learning_rate=0.1,  # 学习率
    n_estimators=100,  # 迭代轮数
    subsample=0.8,  # 采样比例，防止过拟合
    colsample_bytree=0.8,
    use_label_encoder=False,  # 关闭 XGBoost 的 label 编码 (适用于新版本)
    eval_metric='logloss'  # 交叉熵损失
)
clf.fit(X_train, y_train)

# 预测
y_pred_prob = clf.predict(X_test)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'XGBoost 分类准确率: {accuracy:.4f}')
code = m2c.export_to_c(clf)
output_file = "malware_detector.cpp"
with open(output_file, "w") as f:
    f.write(code)
