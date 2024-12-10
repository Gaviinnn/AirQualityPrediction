import os
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from src.data.data_preprocessing import clean_data, feature_engineering
from src.data.transform import split_dataset
from src.models.model import LSTMModel
from src.train.train import train_model
from src.train.evaluate import evaluate_model

if __name__ == "__main__":
    # 加载配置
    with open("./config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_file = config['data']['input_file']
    preprocessed_file = config['data']['preprocessed_file']
    sequence_length = config['model']['sequence_length']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    learning_rate = float(config['model']['learning_rate'])
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    patience = config['model']['patience']
    step_size = config['model']['step_size']
    gamma = config['model']['gamma']
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')

    # 使用绝对路径进行处理，输出文件同样用绝对路径
    intermediate_file = "./data/cleaned_station_data_temp.csv"

    print("Cleaning data...")
    clean_data(input_file, intermediate_file)

    print("Starting feature engineering...")
    feature_engineering(intermediate_file, preprocessed_file)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test, scaler = split_dataset(preprocessed_file, sequence_length=sequence_length, test_ratio=0.2)

    print("Converting NumPy arrays to PyTorch tensors...")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print("Creating TensorDataset and DataLoader...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]  # 特征列数（原始+特征工程）
    output_size = y_train.shape[1] # 5个污染物
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, dropout=dropout).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    try:
        train_model(model, train_loader, criterion, optimizer, epochs, device, patience=patience, step_size=step_size, gamma=gamma)
    except KeyboardInterrupt:
        print("Training interrupted. Cleaning up...")

    best_model_path = "./outputs/model_checkpoint/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model checkpoint.")

    print("Evaluating the model...")
    evaluate_model(model, X_test, y_test, device, scaler)
