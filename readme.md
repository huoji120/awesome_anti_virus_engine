# PE文件恶意软件检测系统

这是一个基于机器学习的PE文件恶意软件检测系统，使用XGBoost算法对PE文件进行分类。

## 功能特点

- 利用PE文件结构特征进行恶意软件检测
- 基于XGBoost机器学习算法
- 提供训练和预测功能
- 输出详细的分类报告和可视化结果

## 系统架构

该系统包含以下组件：

1. **特征提取模块**：C++编写的特征提取器，分析PE文件结构和行为特征
2. **训练模块**：Python编写的模型训练代码，使用XGBoost算法
3. **预测模块**：Python编写的模型推理代码，用于检测未知文件

## 特征集

系统从PE文件中提取以下特征：

1. PE段属性 (是否有配置、调试信息、例外处理、导出、导入等)
2. 导入的DLL库
3. 文件熵
4. 入口点前64字节的归一化值
5. 节区信息 (节区数量、平均熵、最大熵、归一化平均熵、大小比率)
6. 代码段与整个文件的比率
7. 节区数量

## 环境要求

- Python 3.7+
- 依赖包：
  - pandas
  - numpy
  - xgboost
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

安装依赖：

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib
```

## 使用说明

### 1. 准备数据

需要准备两个CSV文件：
- `malware.csv`：恶意软件样本的特征数据
- `whitelist.csv`：正常软件样本的特征数据

这些CSV文件由C++特征提取模块生成。

### 2. 训练模型

运行以下命令进行模型训练：

```bash
python train_model.py
```

训练结果将保存为`xgboost_malware_detector.model`文件，并生成性能评估图表：
- `confusion_matrix.png`：混淆矩阵
- `feature_importance.png`：特征重要性排序

### 3. 预测未知文件

使用训练好的模型预测未知文件：

```bash
python predict.py <csv文件路径1> [csv文件路径2] ...
```

预测结果将保存为`*_predictions.csv`文件。

## 示例

```bash
# 训练模型
python train_model.py

# 预测单个文件
python predict.py unknown_samples.csv

# 批量预测多个文件
python predict.py file1.csv file2.csv file3.csv
```

## 性能指标

在测试数据集上，该系统通常能达到以下性能：

- 准确率：95%+
- 召回率：90%+
- 精确率：92%+
- F1值：91%+

_注意：实际性能可能因训练数据和参数设置而异。_

## 扩展与优化

系统可以进行以下扩展和优化：

1. 添加更多特征，如字符串分析、API调用序列等
2. 尝试其他机器学习算法或深度学习模型
3. 集成多个模型进行综合决策
4. 开发实时监控和检测功能
5. 增加可解释性分析

## License

MIT