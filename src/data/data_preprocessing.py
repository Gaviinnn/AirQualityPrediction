import os
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler

def merge_csv_files(input_folder: str, output_file: str):
    """
    Merge all CSV files in a given folder into a single CSV file.

    :param input_folder: Absolute path to the folder containing CSV files.
    :param output_file: Absolute path to save the merged CSV file.
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    # Collect all CSV files in the folder
    all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not all_files:
        raise ValueError(f"No CSV files found in the folder: {input_folder}")

    # Merge CSV files
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to: {output_file}")

def preprocess_station_data(input_file: str, station_id: int, output_file: str):
    """
    Preprocess data for a specific station.

    :param input_file: Path to the merged CSV file.
    :param station_id: ID of the station to filter (column 'station').
    :param output_file: Path to save the preprocessed data for the station.
    """
    df = pd.read_csv(input_file)

    # 检查 'station' 列是否存在
    if 'station' not in df.columns:
        raise KeyError("The 'station' column is missing in the input file.")

    # 筛选目标站点数据
    filtered_df = df[df['station'] == station_id]
    filtered_df.to_csv(output_file, index=False)
    print(f"Preprocessed station data saved to: {output_file}")


def clean_data(file_path, output_path):
    """
    数据清洗函数，包括处理缺失值、异常值、标准化、删除重复值等操作。

    Args:
        file_path (str): 输入的文件路径
        output_path (str): 清洗后的文件输出路径
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 确保 `date` 列为 datetime 格式
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.set_index('date', inplace=True)

    # 1. 处理缺失值（线性插值）
    data = data.interpolate(method='linear', limit_direction='forward', axis=0)

    # 2. 检测并处理异常值（超过3倍标准差的值替换为均值）
    for col in ['PM10', 'NO_2', 'CO', 'SO_2', 'O_3']:  # 修正列名
        if col in data.columns:  # 确保列存在
            mean, std = data[col].mean(), data[col].std()
            data[col] = data[col].apply(lambda x: mean if abs(x - mean) > 3 * std else x)
        else:
            print(f"Column {col} does not exist in the data!")

    # 3. 数据标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['PM10', 'NO_2', 'CO', 'SO_2', 'O_3']])
    scaled_data = pd.DataFrame(scaled_features, columns=['PM10', 'NO_2', 'CO', 'SO_2', 'O_3'], index=data.index)

    # 4. 删除重复值
    scaled_data = scaled_data.drop_duplicates()

    # 5. 确保时间戳对齐
    scaled_data = scaled_data.resample('h').mean()

    # 保存清洗后的数据
    scaled_data.to_csv(output_path)
    print(f"Cleaned data saved to: {output_path}")


    # 数据生成函数
def create_autoregressive_dataset(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # 输入序列
        y.append(data[i + seq_length])  # 单步目标值
    return np.array(X), np.array(y)


def load_preprocessed_data(file_path, sequence_length, input_size, device):
    try:
        # 加载特征工程后的CSV文件
        data = pd.read_csv(file_path, parse_dates=['date'])
        data.set_index('date', inplace=True)

        # 特征和目标列
        features = data.iloc[:, :-5].values  # 假设目标列是最后5列
        targets = data.iloc[:, -5:].values  # 最后5列为目标

        # 标准化目标列
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        targets = target_scaler.fit_transform(targets)
        with open('target_scaler.pkl', 'wb') as f:
            pickle.dump(target_scaler, f)

        # 转换为PyTorch张量
        features = torch.tensor(features, dtype=torch.float32).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device)

        # 数据集拆分
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(input_file, output_file):
    # 如果 input_file 已经是清洗后的数据，可以直接复制，如果需要额外清洗请在此添加代码
    df = pd.read_csv(input_file, parse_dates=['date'])
    # 假设数据已基本清理好，只需排序和去重，并确保无缺失值
    df = df.sort_values('date').dropna()
    df.to_csv(output_file, index=False)

def feature_engineering(input_file, output_file):
    df = pd.read_csv(input_file, parse_dates=['date'])
    df = df.set_index('date')

    # 对每个污染物添加滞后特征和移动平均特征，示例：1天滞后、7天移动平均
    pollutants = ['PM10', 'NO_2', 'CO', 'SO_2', 'O_3']
    for p in pollutants:
        df[f'{p}_lag1'] = df[p].shift(1)
        df[f'{p}_ma7'] = df[p].rolling(window=7).mean()

    # 去除特征工程导致的缺失值（前几行会有NaN）
    df = df.dropna()

    # 保存特征工程后的数据
    df.to_csv(output_file)