import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# 创建保存目录
os.makedirs("verification/arima", exist_ok=True)
os.makedirs("verification/arima/picture", exist_ok=True)
os.makedirs("verification/arima/对比曲线", exist_ok=True)

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

test_start = datetime(2024, 12, 1)
test_end = datetime(2024, 12, 31)
plot_start = datetime(2024, 11, 1)  # 从11月1日开始的绘图

for item_name in items:
    item_path = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(item_path):
        print(f"{item_path} not found, skipping {item_name}")
        continue

    item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
    if item_df.empty:
        print(f"No data for {item_name}, skipping.")
        continue

    # 划分训练集和测试集
    train_df = item_df[item_df.index < test_start]
    test_df = item_df[(item_df.index >= test_start) & (item_df.index <= test_end)]

    if test_df.empty or len(train_df) < 10:
        print(f"Not enough train or test data for {item_name}, skipping.")
        continue

    # 拟合ARIMA模型
    model = ARIMA(train_df['price'], order=(1,1,1))
    result = model.fit()

    real_values = test_df['price'].values
    test_dates = test_df.index

    # 使用滚动预测：每天预测下一天，然后更新数据和模型
    current_series = train_df['price'].copy()

    predictions = []
    for dt in test_dates:
        fc = result.get_forecast(steps=1)
        pred_val = fc.predicted_mean.iloc[0]
        predictions.append(pred_val)

        # 更新current_series，将真实值加入序列中
        current_series.loc[dt] = test_df.loc[dt, 'price']
        
        # 用更新后的current_series重新拟合模型
        model = ARIMA(current_series, order=(1,1,1))
        result = model.fit()

    predictions = np.array(predictions)

    # 计算误差指标
    mse = mean_squared_error(real_values, predictions)
    mae = mean_absolute_error(real_values, predictions)
    rmse = np.sqrt(mse)

    print(f"{item_name} - December ARIMA forecast metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 保存对比结果
    result_df = pd.DataFrame({
        "date": test_dates,
        "real_price": real_values,
        "predicted_price": predictions
    })
    result_out_path = os.path.join("verification/arima", f"compare_{item_name}.csv")
    result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
    print(f"✅ ARIMA预测结果与真实值已保存: {result_out_path}")

    # 绘制对比图1：仅12月范围
    plt.figure(figsize=(10,5))
    plt.plot(test_dates, real_values, label='Real Price', color='blue')
    plt.plot(test_dates, predictions, label='Predicted Price (ARIMA)', color='red')
    plt.title(f"{item_name} December ARIMA Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    # 在图上显示指标
    text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"
    # 选择一个显示位置，如左上角
    plt.text(test_dates.min(), max(real_values.max(), predictions.max())*0.9, text_str, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path = os.path.join("verification/arima/picture", f"{item_name}_verification_plot_december.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ ARIMA 12月对比图已保存: {picture_path}")

    # 绘制对比图2：从11月开始的版本
    # 在此范围内显示历史数据和预测值（预测值只有12月）
    plt.figure(figsize=(10,5))
    # 显示从11月1日到测试集末尾的数据
    plot_df = item_df.loc[plot_start:test_end]

    plt.plot(plot_df.index, plot_df['price'], label='Historical Price', color='blue')
    # 只在12月显示预测值
    plt.plot(test_dates, predictions, label='Predicted Price (ARIMA)', color='red')
    plt.title(f"{item_name} From Nov to Dec ARIMA Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    # 在图上显示指标
    plt.text(plot_start, max(plot_df['price'].max(), predictions.max())*0.9, text_str, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path_nov = os.path.join("verification/arima/picture", f"{item_name}_verification_plot_from_nov.png")
    plt.savefig(picture_path_nov, dpi=100)
    plt.close()
    print(f"✅ ARIMA 从11月开始的对比图已保存: {picture_path_nov}")

    metrics_path = os.path.join("verification/arima/对比曲线", f"{item_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")
    print(f"✅ ARIMA指标已保存: {metrics_path}")
