import csv
import json
import random
import os
from datetime import datetime

# 读取CSV数据
csv_filename = "csgo_peak_players_20241213.csv"
historic_data = []
with open(csv_filename, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        date_str = row["date"]
        peak = int(row["peak_players"])
        historic_data.append((date_str, peak))

historic_dict = {d: p for d, p in historic_data}

def get_reference_peak():
    return random.choice(historic_data)[1]

def add_noise(value, noise_level=0.1):
    noise = value * noise_level * (2*random.random()-1)
    return max(0, value + noise)

final_stats_file = "data/final_stats.json"
date_record_map = {}
if os.path.exists(final_stats_file):
    with open(final_stats_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_date = record["date"]
            date_record_map[record_date] = record
else:
    print(f"{final_stats_file} not found!")
    exit(1)

def generate_random_record_for_date(date_str):
    def rand_counts():
        pos = random.randint(0, 200)
        neg = random.randint(0, 200)
        total = pos + neg
        ratio = pos/total if total > 0 else 0
        return {"positive_count": pos, "negative_count": neg, "positive_ratio": ratio}

    categories = ["Tournaments", "Version_Updates", "Market_and_Price", "Uncategorized"]
    record = {"date": date_str, "items": {}}
    for cat in categories:
        record[cat] = rand_counts()

    # 随机生成物品数据（可选）
    possible_items = ["ak47", "awp", "m4a1", "knife", "usp", "desert eagle"]
    num_items = random.randint(0,3)
    for _ in range(num_items):
        item = random.choice(possible_items)
        if item not in record["items"]:
            pos = random.randint(0, 50)
            neg = random.randint(0, 50)
            record["items"][item] = {"positive_count": pos, "negative_count": neg}
    return record

for date_str, peak in historic_data:
    if date_str not in date_record_map:
        date_record_map[date_str] = generate_random_record_for_date(date_str)

output_file = "final_complete.json"
with open(output_file, "w", encoding="utf-8") as out_f:
    for date_str, record in sorted(date_record_map.items(), key=lambda x: x[0]):
        ref_peak = get_reference_peak()
        base_peak = 100000
        ratio = base_peak / ref_peak if ref_peak > 0 else 1.0

        for category in ["Tournaments", "Version_Updates", "Market_and_Price", "Uncategorized"]:
            p = record[category]["positive_count"]
            n = record[category]["negative_count"]
            p_scaled = add_noise(p * ratio)
            n_scaled = add_noise(n * ratio)
            record[category]["positive_count"] = int(round(p_scaled))
            record[category]["negative_count"] = int(round(n_scaled))
            total = record[category]["positive_count"] + record[category]["negative_count"]
            if total > 0:
                record[category]["positive_ratio"] = record[category]["positive_count"] / total
            else:
                record[category]["positive_ratio"] = 0.0

        # items处理
        for item_name, counts in record["items"].items():
            p_i = counts["positive_count"]
            n_i = counts["negative_count"]
            p_i_scaled = add_noise(p_i * ratio)
            n_i_scaled = add_noise(n_i * ratio)
            record["items"][item_name]["positive_count"] = int(round(p_i_scaled))
            record["items"][item_name]["negative_count"] = int(round(n_i_scaled))

        # 按指定顺序重构字典，以确保items在最后
        final_record = {
            "date": record["date"],
            "Tournaments": record["Tournaments"],
            "Version_Updates": record["Version_Updates"],
            "Market_and_Price": record["Market_and_Price"],
            "Uncategorized": record["Uncategorized"],
            "items": record["items"]
        }

        out_f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

print("final_complete.json generated with desired key order.")
