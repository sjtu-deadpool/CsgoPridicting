import os
import json
from datetime import datetime, timezone

# ========== 配置部分 ==========

POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "awesome", "buff", "improve", "increase", "boost",
    "positive", "beneficial", "promising", "encouraging", "profitable", "strong", "up",
    "upward", "bullish", "gain", "rise", "prosper", "thriving"
]

NEGATIVE_WORDS = [
    "bad", "awful", "terrible", "nerf", "worse", "down", "loss", "negative", "harmful", "unprofitable",
    "weak", "drop", "downward", "bearish", "decrease", "decline", "detrimental", "risky", "failing", "struggling"
]

# Market_and_Price 相关的物品关键词示例
ITEM_WORDS = ["ak47", "awp", "m4a1", "knife", "usp", "desert eagle"]

MAIN_CATEGORIES = ["Tournaments", "Version_Updates", "Market_and_Price"]

# ========== NLP分析函数 ==========

def analyze_sentiment(text):
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in POSITIVE_WORDS)
    neg_count = sum(text_lower.count(word) for word in NEGATIVE_WORDS)
    return "positive" if pos_count > neg_count else "negative"

def extract_items(text):
    text_lower = text.lower()
    found_items = set()
    for item in ITEM_WORDS:
        if item in text_lower:
            found_items.add(item)
    return found_items

# ========== 处理posts_data并统计结果的函数 ==========

def process_posts(posts_data):
    daily_stats = {}

    for post in posts_data:
        post_date_str = post["post_created_utc"].split(" ")[0]  # 'YYYY-MM-DD'

        if post_date_str not in daily_stats:
            daily_stats[post_date_str] = {
                "Tournaments": {"positive_count": 0, "negative_count": 0},
                "Version_Updates": {"positive_count": 0, "negative_count": 0},
                "Market_and_Price": {"positive_count": 0, "negative_count": 0},
                "Uncategorized": {"positive_count": 0, "negative_count": 0},
                "items": {}
            }

        # 将帖子及其评论整合分析
        all_texts = [post["post_title"], post["post_body"]] + (post.get("comments", []))
        pos_count = 0
        neg_count = 0
        item_sentiment_counts = {}

        for text in all_texts:
            sentiment = analyze_sentiment(text)
            if sentiment == "positive":
                pos_count += 1
            else:
                neg_count += 1

            # 如果帖子包含Market_and_Price分类，统计物品情感
            if "Market_and_Price" in post["categories"]:
                found_items = extract_items(text)
                for it in found_items:
                    if it not in item_sentiment_counts:
                        item_sentiment_counts[it] = {"positive_count": 0, "negative_count": 0}
                    if sentiment == "positive":
                        item_sentiment_counts[it]["positive_count"] += 1
                    else:
                        item_sentiment_counts[it]["negative_count"] += 1

        # 将该帖子的情感计数分配到对应的分类统计中
        if not post["categories"]:
            # 没有分类的帖子放入Uncategorized
            daily_stats[post_date_str]["Uncategorized"]["positive_count"] += pos_count
            daily_stats[post_date_str]["Uncategorized"]["negative_count"] += neg_count
        else:
            found_main_cat = False
            for cat in post["categories"]:
                if cat in MAIN_CATEGORIES:
                    daily_stats[post_date_str][cat]["positive_count"] += pos_count
                    daily_stats[post_date_str][cat]["negative_count"] += neg_count
                    found_main_cat = True
                else:
                    daily_stats[post_date_str]["Uncategorized"]["positive_count"] += pos_count
                    daily_stats[post_date_str]["Uncategorized"]["negative_count"] += neg_count

            # 如果所有分类都不在MAIN_CATEGORIES内，也归为Uncategorized
            if not found_main_cat and post["categories"]:
                daily_stats[post_date_str]["Uncategorized"]["positive_count"] += pos_count
                daily_stats[post_date_str]["Uncategorized"]["negative_count"] += neg_count

        # 合并物品统计结果
        if "Market_and_Price" in post["categories"] and item_sentiment_counts:
            for item_name, counts in item_sentiment_counts.items():
                if item_name not in daily_stats[post_date_str]["items"]:
                    daily_stats[post_date_str]["items"][item_name] = {"positive_count": 0, "negative_count": 0}
                daily_stats[post_date_str]["items"][item_name]["positive_count"] += counts["positive_count"]
                daily_stats[post_date_str]["items"][item_name]["negative_count"] += counts["negative_count"]

    # 计算比例并生成最终JSON数据结构
    final_data = []
    for date_str, stats in daily_stats.items():
        result = {
            "date": date_str
        }
        for cat in MAIN_CATEGORIES + ["Uncategorized"]:
            p = stats[cat]["positive_count"]
            n = stats[cat]["negative_count"]
            total = p + n
            ratio = p / total if total > 0 else 0
            result[cat] = {
                "positive_count": p,
                "negative_count": n,
                "positive_ratio": ratio
            }
        result["items"] = stats["items"]
        final_data.append(result)

    return final_data

def save_final_json(final_data, filename="final_stats.json"):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in final_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_posts_from_data(data_dir="data"):
    categories = ["Tournaments", "Version_Updates", "Market_and_Price", "Uncategorized"]
    posts_map = {}

    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        # 遍历日期文件夹
        for date_folder in os.listdir(cat_dir):
            date_path = os.path.join(cat_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            posts_file = os.path.join(date_path, "posts.json")
            if not os.path.isfile(posts_file):
                continue

            with open(posts_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    post_data = json.loads(line)
                    post_id = post_data["post_id"]
                    if post_id not in posts_map:
                        # 初始化categories列表
                        post_data["categories"] = [cat]
                        posts_map[post_id] = post_data
                    else:
                        # 合并分类
                        if cat not in posts_map[post_id]["categories"]:
                            posts_map[post_id]["categories"].append(cat)

    # 转换为列表
    posts_data = list(posts_map.values())
    return posts_data

if __name__ == "__main__":
    # 从data文件夹加载帖子数据
    posts_data = load_posts_from_data(data_dir="data")
    # 对帖子进行分析和统计
    final_data = process_posts(posts_data)
    # 保存结果为JSON
    save_final_json(final_data, "final_stats.json")
    print("Data NLP analysis complete. Results saved to data/final_stats.json")
