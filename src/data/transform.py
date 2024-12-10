import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def split_dataset(file_path, sequence_length=24, test_ratio=0.2):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.set_index('date')

    # 假设原始5个污染物列依然存在：PM10, NO_2, CO, SO_2, O_3
    # 特征工程后增加了更多列，如 PM10_lag1, PM10_ma7 等
    # 我们的特征包括所有列(不只这5列), 但目标仍是这5个污染物当天的值。
    # 假设目标列仍然是 [PM10, NO_2, CO, SO_2, O_3] 原始列
    target_cols = ['PM10', 'NO_2', 'CO', 'SO_2', 'O_3']
    feature_cols = df.columns.tolist()  # 包含所有特征列（原始 + 特征工程列）

    # 确保目标列在特征中
    # 特征和目标都包含在DataFrame中，但我们需要分开处理
    # 特征包含所有列，目标是这5列。训练时我们用过去24小时的所有特征预测下一个时刻这5列的值
    values = df.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)

    # 找出target列的索引位置，用于在y中提取目标数据
    target_indices = [feature_cols.index(c) for c in target_cols]

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length, :])  # 所有特征
        y.append(scaled[i + sequence_length, target_indices])  # 对应的5个污染物作为目标

    X = np.array(X)
    y = np.array(y)

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler
