import praw  
import pandas as pd  
import time  
from datetime import datetime  

# Reddit API credentials  
reddit = praw.Reddit(  
    client_id="pJ4fBX-S5S7rM13M2ZvavQ",                    # 替换为你的 Client ID  
    client_secret="sPwEBHdPx_jcgXLhKg_uHWcAkrTMRQ",         # 替换为你的 Client Secret  
    user_agent="csgo predict"  
)

def fetch_from_subreddit(subreddit_name="GlobalOffensive", limit=100):  
    """  
    Fetch Reddit posts and comments from a subreddit.  

    Args:  
        subreddit_name (str): Target subreddit.  
        limit (int): Number of posts to fetch during each batch.  

    Returns:  
        list: List of posts, including titles, bodies, comments, etc.  
    """  
    subreddit = reddit.subreddit(subreddit_name)  
    posts_info = []  

    # Get new posts (limit specifies the number of posts per batch)  
    for post in subreddit.new(limit=limit):  
        post_details = {  
            "post_id": post.id,  
            "post_title": post.title,  
            "post_body": post.selftext if post.selftext else "N/A",  
            "post_score": post.score,  
            "post_author": str(post.author),  
            "post_created_utc": datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),  
            "comment_count": post.num_comments,  
            "comments": [],  
        }  

        # Fetch up to 20 comments for each post  
        post.comments.replace_more(limit=0)  
        comments = [comment.body for comment in post.comments.list()[:20]]  
        post_details["comments"] = comments  

        posts_info.append(post_details)  

    return posts_info  


def save_to_csv(data, batch_num):  
    """  
    Save the fetched data to a CSV file.  

    Args:  
        data (list): List of posts with metadata.  
        batch_num (int): The batch number to differentiate multiple files.  

    Returns:  
        str: Filename of the saved file.  
    """  
    filename = f"globaloffensive_data_{datetime.now().strftime('%Y%m%d')}_batch{batch_num}.csv"  
    pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8')  
    print(f"Saved batch {batch_num} to {filename}")  
    return filename  


def crawl_subreddit(subreddit_name="GlobalOffensive"):  
    """  
    Continuously crawl a subreddit and handle rate limits.  

    Args:  
        subreddit_name (str): The subreddit to crawl (default is "GlobalOffensive").  
    """  
    batch_num = 1  

    while True:  
        try:  
            print(f"Starting batch {batch_num}...")  

            # Fetch 100 posts per request  
            posts = fetch_from_subreddit(subreddit_name=subreddit_name, limit=100)  

            if not posts:  
                print("No more posts available. Stopping.")  
                break  

            # Save the batch to a CSV file  
            save_to_csv(posts, batch_num)  
            batch_num += 1  

            # Wait 60 seconds before making the next batch request  
            print("Waiting 60 seconds before fetching the next batch...")  
            time.sleep(60)  

        except Exception as e:  
            print(f"Encountered an error: {e}. Waiting 10 minutes before retrying...")  
            time.sleep(600)  # Sleep 10 minutes and retry  


# Start the crawling process  
crawl_subreddit()