import os
import json
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 确保输出目录存在
os.makedirs("item_smooth_prices", exist_ok=True)
os.makedirs("consequence/arima", exist_ok=True)
os.makedirs("consequence/arima/picture", exist_ok=True)

price_dir = "item/prices"
items = [
    "AK-47__Redline_(Battle-Scarred)",
    "AK-47__Blue_Laminate_(Field-Tested)",
    "AK-47__Asiimov_(Factory_New)",
    "AK-47__Nightwish_(Factory_New)"
]

# 1. 平滑和插值处理物品价格数据
for item_name in items:
    price_file = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(price_file):
        print(f"{price_file} not found, skipping.")
        continue
    
    df = pd.read_csv(price_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # 去掉无法解析日期的行
    df = df.set_index('date').sort_index()

    if df.empty:
        print(f"No data for {item_name}, skipping.")
        continue

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)
    df['price'] = df['price'].interpolate(method='linear', limit_direction='both')
    df.index.name = "date"

    smooth_path = os.path.join("item_smooth_prices", f"{item_name}.csv")
    df.to_csv(smooth_path, index=True, date_format='%Y-%m-%d')
    print(f"✅ 平滑价格数据已保存: {smooth_path}")

# 2. 读取 data_complete.json
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

smooth_dir = "item_smooth_prices"
forecast_steps = 7

for item_name in items:
    smooth_item_path = os.path.join(smooth_dir, f"{item_name}.csv")
    if not os.path.exists(smooth_item_path):
        print(f"{smooth_item_path} not found, cannot proceed with ARIMA forecast for {item_name}.")
        continue

    item_df = pd.read_csv(smooth_item_path, parse_dates=["date"], index_col="date")

    if df_data.empty or item_df.empty:
        print(f"No overlapping data for {item_name}, skipping forecast.")
        continue

    common_start = max(df_data.index.min(), item_df.index.min())
    common_end = min(df_data.index.max(), item_df.index.max())

    if common_start > common_end:
        print(f"No overlapping date range between data_complete and item prices for {item_name}.")
        continue

    df_data_item = df_data.loc[common_start:common_end]
    item_df = item_df.loc[common_start:common_end]

    if item_df['price'].isna().all():
        print(f"All prices are NaN for {item_name} in overlapping range, skipping.")
        continue

    # 使用ARIMA进行预测
    y = item_df['price'].fillna(method='ffill')  # 确保无缺失值
    model = sm.tsa.ARIMA(y, order=(1,1,1))
    result = model.fit()
    print(result.summary())

    forecast = result.forecast(steps=forecast_steps)
    forecast_df = pd.DataFrame({
        "date": pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=forecast_steps),
        "forecast_price": forecast
    })
    forecast_df.set_index('date', inplace=True)

    # 输出 future_前缀_ 文件(仅未来数据)
    future_out_path = os.path.join("consequence/arima", f"future_{item_name}.csv")
    forecast_df.to_csv(future_out_path, date_format='%Y-%m-%d')
    print(f"✅ 未来预测数据已保存: {future_out_path}")

    # 输出 complete_前缀_ 文件(历史+未来数据)
    # 将历史数据与预测数据拼接
    complete_df = pd.concat([y, forecast_df['forecast_price']], axis=1)
    # 历史数据列名设为 "price", 未来列为 "forecast_price"
    complete_df.rename(columns={0:"forecast_price"}, inplace=True)
    # 历史数据本来在y中就是price列
    # 将历史部分的forecast_price为空值填NA或0
    complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
    complete_out_path = os.path.join("consequence/arima", f"complete_{item_name}.csv")
    complete_df.to_csv(complete_out_path, date_format='%Y-%m-%d')
    print(f"✅ 历史+未来数据已保存: {complete_out_path}")

    # 作图: 历史数据为蓝色线, 预测部分为红色线
    plt.figure(figsize=(10,5))
    plt.plot(y.index, y, label='Historical Price', color='blue')
    plt.plot(forecast_df.index, forecast_df['forecast_price'], label='Forecast Price', color='red')
    plt.title(f"Price Forecast for {item_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    picture_dir = "consequence/arima/picture"
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{item_name}_forecast_plot.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ 图表已保存: {picture_path}")
