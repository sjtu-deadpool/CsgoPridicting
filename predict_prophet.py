import os
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 创建输出目录（类似于CGAN的结构）
os.makedirs("verification/prophet_yearly_only", exist_ok=True)
os.makedirs("verification/prophet_yearly_only/picture", exist_ok=True)
os.makedirs("verification/prophet_yearly_only/对比曲线", exist_ok=True)

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
        records.append(record_date)

df_data = pd.DataFrame(records, columns=["date"])
df_data = df_data.set_index("date").sort_index()

# 我们假设历史数据至少到2024-12-13
end_date = datetime(2024, 12, 13)
start_pred = end_date + timedelta(days=1)  # 2024-12-14开始预测
future_days = 365  # 预测一年
end_pred = start_pred + timedelta(days=future_days-1)  # 2025-12-13

for item_name in items:
    item_path = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(item_path):
        print(f"{item_path} not found, skipping {item_name}")
        continue

    item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
    if item_df.empty:
        print(f"No data for {item_name}, skipping Prophet forecast.")
        continue

    # 确保数据截止至end_date
    item_df = item_df[item_df.index <= end_date]
    if item_df.empty:
        print(f"No data up to {end_date} for {item_name}, skipping.")
        continue

    # Prophet需要ds,y列
    df_prophet = item_df.copy()
    df_prophet = df_prophet.rename(columns={"price": "y"})
    df_prophet['ds'] = df_prophet.index
    df_prophet = df_prophet[['ds', 'y']]

    # 初始化仅使用年度季节性的Prophet模型
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.fit(df_prophet)

    # 创建未来日期数据
    future = model.make_future_dataframe(periods=future_days, freq='D')
    forecast = model.predict(future)

    # 截取预测区间（2024-12-14到2025-12-13）
    forecast_period = forecast[(forecast['ds'] >= start_pred) & (forecast['ds'] <= end_pred)]

    # 保存未来预测数据
    future_path = os.path.join("verification/prophet_yearly_only", f"future_prophet_yearly_{item_name}.csv")
    forecast_period.to_csv(future_path, index=False, date_format='%Y-%m-%d')
    print(f"✅ Prophet未来一年预测已保存: {future_path}")

    # 构建完整序列（历史+未来）
    complete_df = pd.concat([df_prophet.set_index('ds')[['y']], 
                             forecast_period.set_index('ds')[['yhat']]], axis=1)
    complete_df = complete_df.rename(columns={'y':'price','yhat':'forecast_price'})
    complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
    complete_path = os.path.join("verification/prophet_yearly_only", f"complete_prophet_yearly_{item_name}.csv")
    complete_df.to_csv(complete_path, date_format='%Y-%m-%d')
    print(f"✅ Prophet完整序列已保存: {complete_path}")

    # 绘图：显示历史数据和未来一年的预测
    plt.figure(figsize=(10,5))
    plt.plot(item_df.index, item_df['price'], label='Historical Price', color='blue')
    plt.plot(forecast_period['ds'], forecast_period['yhat'], label='Prophet Forecast (Yearly Only)', color='green')
    plt.title(f"Prophet Yearly-Only Seasonality Forecast from {start_pred.date()} for 1 Year\n{item_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    picture_path = os.path.join("verification/prophet_yearly_only/picture", f"{item_name}_prophet_yearly_forecast_plot.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ Prophet预测图已保存: {picture_path}")

    # 如果需要查看seasonality和趋势成分
    # comp_fig = model.plot_components(forecast)
    # comp_fig.savefig(os.path.join("verification/prophet_yearly_only/picture", f"{item_name}_prophet_yearly_components.png"), dpi=100)
    # plt.close(comp_fig)

    # 本例中未计算误差指标，因为我们没有对比测试集，如果有未来真实值可以对齐再计算
    # 若有真实对比数据，也可类似之前计算MSE、MAE、RMSE并保存
