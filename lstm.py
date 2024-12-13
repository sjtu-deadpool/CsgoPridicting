import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 确保输出目录存在
os.makedirs("consequence/lstm", exist_ok=True)
os.makedirs("consequence/lstm/picture", exist_ok=True)

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
        peak_players = record.get("peak_players", 0)
        trends = record.get("trends", {})
        csgo_trend = trends.get("CS:GO", 0)
        records.append([record_date, peak_players, csgo_trend])

df_data = pd.DataFrame(records, columns=["date","peak_players","csgo_trend"])
df_data = df_data.set_index("date").sort_index()

forecast_steps = 7
window_size = 30  # 使用过去30天的数据预测下一天

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

    if df_data.empty or item_df.empty:
        print(f"No overlapping data for {item_name}, skipping LSTM forecast.")
        continue

    common_start = max(df_data.index.min(), item_df.index.min())
    common_end = min(df_data.index.max(), item_df.index.max())

    if common_start > common_end:
        print(f"No overlapping date range between data_complete and item prices for {item_name}.")
        continue

    item_df = item_df.loc[common_start:common_end]

    if item_df.empty:
        print(f"No data after intersection for {item_name}, skipping.")
        continue

    # 取价格序列
    series = item_df['price'].values.reshape(-1,1)

    # 数据归一化(可选)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_series = scaler.fit_transform(series)

    # 构建监督学习数据集
    X, Y = create_dataset(scaled_series, window_size)
    if len(X) == 0:
        print(f"Not enough data to create a dataset for {item_name}, skipping.")
        continue

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X, Y, epochs=10, batch_size=32, verbose=1)

    # 预测未来 n 天
    # 使用最近window_size天的数据递归预测未来
    last_window = scaled_series[-window_size:]
    current_input = last_window.copy().reshape(1, window_size, 1)
    future_preds = []

    for _ in range(forecast_steps):
        pred = model.predict(current_input, verbose=0)
        future_preds.append(pred[0][0])
        # 将预测值加入当前窗口尾部
        # [[[pred[0][0]]]] 以保持(1,1,1)的形状，匹配current_input[:,1:,:]的shape
        current_input = np.append(current_input[:,1:,:], [[[pred[0][0]]]], axis=1)

    # 反归一化预测值
    future_preds = np.array(future_preds).reshape(-1,1)
    future_preds_inv = scaler.inverse_transform(future_preds)

    # 生成未来预测日期索引
    future_dates = pd.date_range(start=item_df.index.max() + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=["forecast_price"])

    # 保存future_lstm_文件
    future_path = os.path.join("consequence/lstm", f"future_lstm_{item_name}.csv")
    forecast_df.to_csv(future_path, date_format='%Y-%m-%d')
    print(f"✅ LSTM未来预测已保存: {future_path}")

    # 构建complete_lstm_文件（历史+未来）
    complete_df = pd.concat([item_df[['price']], forecast_df], axis=1)
    # 历史部分没有forecast_price填充''
    complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
    complete_path = os.path.join("consequence/lstm", f"complete_lstm_{item_name}.csv")
    complete_df.to_csv(complete_path, date_format='%Y-%m-%d')
    print(f"✅ LSTM完整序列已保存: {complete_path}")

    # 绘图，历史为蓝色，预测为绿色
    plt.figure(figsize=(10,5))
    plt.plot(item_df.index, item_df['price'], label='Historical Price', color='blue')
    plt.plot(forecast_df.index, forecast_df['forecast_price'], label='LSTM Forecast', color='green')
    plt.title(f"LSTM Price Forecast for {item_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    picture_dir = "consequence/lstm/picture"
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{item_name}_lstm_forecast_plot.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ LSTM预测图已保存: {picture_path}")
