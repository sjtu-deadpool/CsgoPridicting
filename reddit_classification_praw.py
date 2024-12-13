import praw
import os
import json
import pandas as pd
from datetime import datetime, timezone

# Reddit API credentials
reddit = praw.Reddit(
    client_id="pJ4fBX-S5S7rM13M2ZvavQ",
    client_secret="sPwEBHdPx_jcgXLhKg_uHWcAkrTMRQ",
    user_agent="csgo predict"
)

KEYWORDS = {
    "Tournaments": [
        "major", "tournament", "competition", "match", "team", "players", "league",
        "qualifier", "iem", "esl", "blast", "pgl", "finals", "playoff", "champion"
    ],
    "Version_Updates": [
        "update", "patch", "release", "fix", "changelog", "adjustment", "changes",
        "buff", "improvement", "enhance", "boost", "upgrade", "nerf", "reduce", "downgrade",
        "new map", "map update", "map changes", "map release", "map pool"
    ],
    "Market_and_Price": [
        "price", "cost", "value", "skin", "trade", "market", "deal", "worth", "investment",
        "new case", "case update", "case release", "new item", "new skin", "new weapon", "exclusive item"
    ]
}

SPECIAL_CATEGORIES = ["new case", "case release", "new item", "new skin", "new weapon", "exclusive item"]

processed_ids = set()

def classify_post(title):
    categories = []
    for category, keywords in KEYWORDS.items():
        if any(keyword.lower() in title.lower() for keyword in keywords):
            categories.append(category)

    # Special handling for certain keywords
    if any(keyword.lower() in title.lower() for keyword in SPECIAL_CATEGORIES):
        if "Market_and_Price" not in categories:
            categories.append("Market_and_Price")

    return categories

def fetch_comments(post):
    try:
        post.comments.replace_more(limit=0)
        comments = [comment.body for comment in post.comments.list()[:20]]
        return comments
    except Exception as e:
        print(f"Error fetching comments for post {post.id}: {e}")
        return []

def save_to_json(data, folder, date):
    directory = f"data/{folder}/{date}"
    os.makedirs(directory, exist_ok=True)

    filename = f"{directory}/posts.json"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def save_daily_stats(all_stats):
    # all_stats 是一个按日期为key的字典，每个value是统计数据的dict
    filename = "daily_stats.csv"
    records = []
    for date, stats in all_stats.items():
        # stats本身已包含计数结果
        stats["date"] = date
        records.append(stats)
    df = pd.DataFrame(records)
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(filename, index=False, encoding="utf-8")

def fetch_and_classify_posts(subreddit_name="GlobalOffensive"):
    subreddit = reddit.subreddit(subreddit_name)

    # 使用一个字典，以post_date为key来统计
    daily_stats_map = {}  # { "YYYY-MM-DD": {"total_posts":..., "Tournaments":..., ...}, ... }

    for post in subreddit.new(limit=None):
        if post.id in processed_ids:
            continue
        processed_ids.add(post.id)

        # 获取帖子创建日期
        post_datetime = datetime.fromtimestamp(post.created_utc, timezone.utc)
        post_date = post_datetime.strftime('%Y-%m-%d')

        # 初始化该日期的统计数据（如不存在）
        if post_date not in daily_stats_map:
            daily_stats_map[post_date] = {
                "total_posts": 0,
                "Tournaments": 0,
                "Version_Updates": 0,
                "Market_and_Price": 0,
                "Uncategorized": 0
            }

        daily_stats_map[post_date]["total_posts"] += 1

        categories = classify_post(post.title)

        post_data = {
            "post_id": post.id,
            "post_title": post.title,
            "post_body": post.selftext if post.selftext else "N/A",
            "post_score": post.score,
            "post_author": str(post.author),
            "post_created_utc": post_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "comment_count": post.num_comments,
        }

        if categories:
            post_data["comments"] = fetch_comments(post)
            # 对所有分类进行计数和存储
            for category in categories:
                save_to_json(post_data, category, post_date)
                daily_stats_map[post_date][category] += 1
        else:
            save_to_json(post_data, "Uncategorized", post_date)
            daily_stats_map[post_date]["Uncategorized"] += 1

    return daily_stats_map

# 一次性运行，从最新的帖子一直往历史抓取，并根据帖子实际发布日期分类存储
all_stats = fetch_and_classify_posts(subreddit_name="GlobalOffensive")
save_daily_stats(all_stats)

print("Data collection complete.")
for date, stats in all_stats.items():
    print(f"{date}: {stats}")
