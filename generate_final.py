import os
import json
import random
import pandas as pd
from datetime import datetime

# 读取 final_complete.json
final_complete_file = "final_complete.json"
final_data = {}
if os.path.exists(final_complete_file):
    with open(final_complete_file, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            record = json.loads(line)
            date_str = record["date"]
            final_data[date_str] = record
else:
    print("final_complete.json not found!")
    exit(1)

# 找出所有出现过的物品名称
all_items = set()
for d in final_data.values():
    for it in d["items"].keys():
        all_items.add(it)
all_items = sorted(all_items)

# 创建DataFrame
# 列结构：
# Tournaments_positive_count, Tournaments_negative_count, Tournaments_positive_ratio, 
# Version_Updates_positive_count, Version_Updates_negative_count, Version_Updates_positive_ratio, ...
# Market_and_Price_..., Uncategorized_...
# 对items每个item生成 item_<name>_positive_count, item_<name>_negative_count
categories = ["Tournaments", "Version_Updates", "Market_and_Price", "Uncategorized"]

columns = []
for cat in categories:
    columns += [f"{cat}_positive_count", f"{cat}_negative_count", f"{cat}_positive_ratio"]
for it in all_items:
    columns += [f"item_{it}_positive_count", f"item_{it}_negative_count"]

# 将final_data转化为DataFrame
rows = []
for date_str, rd in final_data.items():
    row = {}
    for cat in categories:
        row[f"{cat}_positive_count"] = rd[cat]["positive_count"]
        row[f"{cat}_negative_count"] = rd[cat]["negative_count"]
        row[f"{cat}_positive_ratio"] = rd[cat]["positive_ratio"]
    # items
    for it in all_items:
        if it in rd["items"]:
            row[f"item_{it}_positive_count"] = rd["items"][it]["positive_count"]
            row[f"item_{it}_negative_count"] = rd["items"][it]["negative_count"]
        else:
            row[f"item_{it}_positive_count"] = None
            row[f"item_{it}_negative_count"] = None
    rows.append((date_str, row))

df = pd.DataFrame([r for _,r in rows], index=[datetime.strptime(d, "%Y-%m-%d") for d,_ in rows], columns=columns)

# 找出日期范围
start_date = datetime(2012, 7, 1)
max_date_in_final = df.index.max()
if pd.isnull(max_date_in_final):
    max_date_in_final = start_date  # 如果没有数据则设为start_date
full_range = pd.date_range(start_date, max_date_in_final, freq='D')

# 对final数据也进行插值
df = df.reindex(full_range)
# 将字符串列转为float，如果有的话（这里应该全是数值）
df = df.astype(float)
df = df.interpolate(method='linear', limit_direction='both')

# 插值后NaN填0
df = df.fillna(0)

# 插值完成后再将DataFrame转换回字典格式
interp_final_data = {}
for dt in df.index:
    date_str = dt.strftime("%Y-%m-%d")
    record = {"date": date_str, "items": {}}
    for cat in categories:
        p = df.loc[dt, f"{cat}_positive_count"]
        n = df.loc[dt, f"{cat}_negative_count"]
        total = p+n
        # 如果需要重新计算ratio确保合理性：
        ratio = p/total if total > 0 else 0.0
        record[cat] = {
            "positive_count": int(round(p)),
            "negative_count": int(round(n)),
            "positive_ratio": ratio
        }
    # items
    for it in all_items:
        p_i = df.loc[dt, f"item_{it}_positive_count"]
        n_i = df.loc[dt, f"item_{it}_negative_count"]
        if p_i==0 and n_i==0:
            # 如果插值后全为0，可以选择不写入items或者写入空
            continue
        record["items"][it] = {
            "positive_count": int(round(p_i)),
            "negative_count": int(round(n_i))
        }
    interp_final_data[date_str] = record

# 接下来读 peak_players 和 trend_data，并对其插值，与之前代码类似
peak_file = "csgo_peak_players_20241213.csv"
peak_df = pd.read_csv(peak_file, parse_dates=["date"], index_col="date")
peak_df = peak_df[peak_df.index >= start_date]
trend_file = "csgo_trend_data.csv"
trend_df = pd.read_csv(trend_file, parse_dates=["date"], index_col="date")
trend_df = trend_df[trend_df.index >= start_date]

peak_df = peak_df.reindex(full_range)
trend_df = trend_df.reindex(full_range)

peak_df["peak_players"] = peak_df["peak_players"].interpolate(method='linear', limit_direction='both')

for col in trend_df.columns:
    trend_df[col] = pd.to_numeric(trend_df[col], errors='coerce')
trend_df = trend_df.interpolate(method='linear', limit_direction='both')
trend_df = trend_df.fillna(0)

# 将所有数据合并，输出 data_complete.json
output_file = "data_complete.json"
with open(output_file, "w", encoding="utf-8") as out_f:
    for dt in full_range:
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in interp_final_data:
            record = interp_final_data[date_str]
        else:
            # 如果日期在final_data中不存在，则生成空记录
            record = {
                "date": date_str,
                "Tournaments": {"positive_count":0, "negative_count":0, "positive_ratio":0.0},
                "Version_Updates": {"positive_count":0, "negative_count":0, "positive_ratio":0.0},
                "Market_and_Price": {"positive_count":0, "negative_count":0, "positive_ratio":0.0},
                "Uncategorized": {"positive_count":0, "negative_count":0, "positive_ratio":0.0},
                "items": {}
            }

        # peak_players
        peak_val = peak_df.loc[dt, "peak_players"]
        if pd.isna(peak_val):
            peak_val = 0
        record["peak_players"] = int(round(peak_val))

        # trends
        trend_data_for_day = trend_df.loc[dt]
        trend_dict = trend_data_for_day.to_dict()
        for k,v in trend_dict.items():
            if pd.isna(v):
                trend_dict[k] = 0
        record["trends"] = trend_dict

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("data_complete.json generated with linear interpolation for final data, peak players and trend data.")
