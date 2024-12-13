import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 创建保存目录
os.makedirs("verification/lstm", exist_ok=True)
os.makedirs("verification/lstm/picture", exist_ok=True)
os.makedirs("verification/lstm/对比曲线", exist_ok=True)

price_dir = "item_smooth_prices"
items = [
    "AK-47__Redline_(Battle-Scarred)",
    "AK-47__Blue_Laminate_(Field-Tested)",
    "AK-47__Asiimov_(Factory_New)",
    "AK-47__Nightwish_(Factory_New)"
]

data_complete_file = "data_complete.json"
records = []
with open(data_complete_file, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        record = json.loads(line)
        record_date = datetime.strptime(record["date"], "%Y-%m-%d")
        records.append([record_date])

df_data = pd.DataFrame(records, columns=["date"])
df_data = df_data.set_index("date").sort_index()

# 定义日期范围
test_start = datetime(2024, 12, 1)
test_end = datetime(2024, 12, 31)
plot_start = datetime(2024, 11, 1)  # 从11月1日开始进行更广范围的绘图

window_size = 30

def create_dataset(series, window_size=30):
    X, Y = [], []
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

for item_name in items:
    item_path = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(item_path):
        print(f"{item_path} not found, skipping {item_name}")
        continue

    item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
    if item_df.empty:
        print(f"No data for {item_name}, skipping.")
        continue

    # 将12月的数据作为测试集，其余作为训练集
    train_df = item_df[item_df.index < test_start]
    test_df = item_df[(item_df.index >= test_start) & (item_df.index <= test_end)]

    if len(train_df) < window_size or test_df.empty:
        print(f"Not enough train or test data for {item_name}, skipping.")
        continue

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0,1))
    train_series = scaler.fit_transform(train_df['price'].values.reshape(-1,1))

    X_train, Y_train = create_dataset(train_series, window_size)
    if len(X_train) == 0:
        print(f"Not enough training data after windowing for {item_name}, skipping.")
        continue

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

    # 对测试集进行滚动预测
    full_series = np.concatenate([train_series, scaler.transform(test_df['price'].values.reshape(-1,1))], axis=0)
    test_start_idx = len(train_series)
    real_values = test_df['price'].values
    test_dates = test_df.index

    # 用训练集中最后window_size天作为初始窗口
    cond = full_series[test_start_idx - window_size:test_start_idx].copy().reshape(1, window_size, 1)

    predictions = []
    for i in range(len(test_df)):
        pred = model.predict(cond, verbose=0)
        predictions.append(pred[0][0])
        # 使用预测值滚动更新cond
        # 注意这里改成 [[[pred[0][0]]]] 保证是(1,1,1)
        cond = np.append(cond[:,1:,:], [[[pred[0][0]]]], axis=1)

    predictions = np.array(predictions).reshape(-1,1)
    predictions_inv = scaler.inverse_transform(predictions)

    # 计算误差指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(real_values, predictions_inv[:,0])
    mae = mean_absolute_error(real_values, predictions_inv[:,0])
    rmse = np.sqrt(mse)

    print(f"{item_name} - December LSTM forecast metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 保存对比结果
    result_df = pd.DataFrame({
        "date": test_dates,
        "real_price": real_values,
        "predicted_price": predictions_inv[:,0]
    })
    result_out_path = os.path.join("verification/lstm", f"compare_{item_name}.csv")
    result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
    print(f"✅ LSTM预测结果与真实值已保存: {result_out_path}")

    text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"

    # 绘制对比图1：仅12月范围
    plt.figure(figsize=(10,5))
    plt.plot(test_dates, real_values, label='Real Price', color='blue')
    plt.plot(test_dates, predictions_inv[:,0], label='Predicted Price', color='red')
    plt.title(f"{item_name} December LSTM Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.text(test_dates.min(), max(real_values.max(), predictions_inv[:,0].max())*0.9, text_str,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path = os.path.join("verification/lstm/picture", f"{item_name}_verification_plot_december.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ LSTM 12月对比图已保存: {picture_path}")

    # 绘制对比图2：从11月开始
    plt.figure(figsize=(10,5))
    extended_plot_df = item_df.loc[(item_df.index >= plot_start) & (item_df.index <= test_end)]

    plt.plot(extended_plot_df.index, extended_plot_df['price'], label='Historical Price', color='blue')
    plt.plot(test_dates, predictions_inv[:,0], label='Predicted Price', color='red')
    plt.title(f"{item_name} From Nov to Dec LSTM Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.text(plot_start, max(extended_plot_df['price'].max(), predictions_inv[:,0].max())*0.9, text_str,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path_nov = os.path.join("verification/lstm/picture", f"{item_name}_verification_plot_from_nov.png")
    plt.savefig(picture_path_nov, dpi=100)
    plt.close()
    print(f"✅ LSTM从11月开始的对比图已保存: {picture_path_nov}")

    metrics_path = os.path.join("verification/lstm/对比曲线", f"{item_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")
    print(f"✅ LSTM指标已保存: {metrics_path}")
