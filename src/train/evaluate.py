import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, device, scaler):
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_t).cpu().numpy()
        actuals = y_test

    # 反归一化
    predictions_inv = scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], scaler.scale_.shape[0]-predictions.shape[1]))]))[:, :predictions.shape[1]]
    actuals_inv = scaler.inverse_transform(np.hstack([actuals, np.zeros((actuals.shape[0], scaler.scale_.shape[0]-actuals.shape[1]))]))[:, :actuals.shape[1]]

    # 加权计算综合污染指数
    weights = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    predicted_aqi = (predictions_inv * weights).sum(axis=1)
    actual_aqi = (actuals_inv * weights).sum(axis=1)

    # 可视化结果
    plt.figure(figsize=(10,6))
    plt.plot(range(len(actual_aqi)), actual_aqi, label="Actual AQI")
    plt.plot(range(len(predicted_aqi)), predicted_aqi, label="Predicted AQI")
    plt.title("Air Quality Index (AQI): Predicted vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("AQI")
    plt.legend()
    plt.savefig("./outputs/plots/aqi_prediction.png")
    plt.show()

    # 输出每种污染物的MSE
    mse_per_pollutant = mean_squared_error(actuals_inv, predictions_inv, multioutput='raw_values')
    print("MSE per pollutant:", mse_per_pollutant)

    # 计算AQI的RMSE
    aqi_rmse = np.sqrt(((predicted_aqi - actual_aqi) ** 2).mean())
    print(f"AQI RMSE: {aqi_rmse:.4f}")

    # 保存预测结果
    results = np.hstack([predictions_inv, predicted_aqi.reshape(-1,1)])
    columns = ["PM10_pred","NO2_pred","CO_pred","SO2_pred","O3_pred","AQI_pred"]
    pd.DataFrame(results, columns=columns).to_csv("./outputs/predictions/predictions.csv", index=False)
