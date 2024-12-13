import time  
from pytrends.request import TrendReq  
import pandas as pd  


def fetch_trend_data_with_backoff(keywords, timeframe="all", region="", max_retries=5):  
    """  
    Fetch Google Trends data for multiple keywords with exponential backoff.  

    Args:  
        keywords (list): A list of topics to track (e.g., ["CS:GO", "major"]).  
        timeframe (str): The timeframe to query (e.g., "today 12-m", "all").  
        region (str): The country-specific interest (default is global).  
        max_retries (int): Maximum number of retries after failures.  

    Returns:  
        pd.DataFrame: A DataFrame containing trends data for all keywords.  
    """  
    # Initialize Pytrends  
    pytrends = TrendReq(hl="en-US", tz=360)  

    all_trends = pd.DataFrame()  

    for keyword in keywords:  
        print(f"Fetching trend data for keyword: {keyword}")  
        
        # Initialize retry variables  
        retries = 0  
        wait_time = 10  # Initial delay in seconds  

        while retries <= max_retries:  
            try:  
                # Build a payload for the keyword  
                pytrends.build_payload([keyword], timeframe=timeframe, geo=region)  
                
                # Fetch interest over time  
                data = pytrends.interest_over_time()  
                
                if data.empty:  
                    print(f"No trend data found for the keyword: {keyword}")  
                    break  
                
                # Remove unrelated columns (isPartial)  
                data = data.drop(columns=["isPartial"])  
                
                # Add the keyword's data to the main DataFrame  
                if all_trends.empty:  
                    all_trends = data.rename(columns={keyword: keyword})  
                else:  
                    all_trends = all_trends.join(data.rename(columns={keyword: keyword}), how="outer")  
                
                # If the query succeeds, break the retry loop  
                break  
            
            except Exception as e:  
                retries += 1  
                print(f"Error fetching data for {keyword}: {e}")  
                
                if retries > max_retries:  
                    print(f"Maximum retries reached for {keyword}. Skipping...")  
                    break  
                
                # Wait for an exponentially increasing amount of time  
                print(f"Retrying in {wait_time} seconds...")  
                time.sleep(wait_time)  
                wait_time *= 2  # Double the wait time with each retry  
                
    return all_trends  


# Example usage  
if __name__ == "__main__":  
    keywords = [  
        "CS:GO", "Counter-Strike", "Global Offensive", "CS2", "Valve CS",  
        "CS:GO Major", "Major Championship", "tournament", "IEM Katowice", "ESL One",  
        "PGL Major", "BLAST Premier", "DreamHack", "Faceit Major", "ESEA league",  
        "CS:GO Finals", "CS:GO Playoffs", "Valve Major", "CS:GO esports", "CS2 update",  
        "CS:GO qualifiers", "CS:GO playoff match", "CS:GO grand finals", "CS:GO patch",  
        "Operation Riptide", "Operation Broken Fang", "Operation Hydra", "Operation Wildfire",  
        "CS:GO skins", "CS:GO trade", "CS:GO market", "CS:GO knives", "CS:GO case",   
        "CS:GO trading platform", "CS:GO keys", "CS:GO AK-47 skins", "CS:GO AWP skins",  
        "CS:GO knives market", "CS:GO sticker", "CS:GO graffiti", "CS:GO VAC ban",   
        "CS:GO hacking", "CS:GO ban detection", "CS2 VAC system", "CS:GO pro players",  
        "CS:GO best team", "CS:GO news", "CS:GO highlights", "CS:GO funny moments",  
        "CS:GO maps", "CS:GO Mirage", "CS:GO Dust 2", "CS:GO Inferno"  
    ]  
    
    trend_data = fetch_trend_data_with_backoff(keywords, timeframe="all")  
    
    if not trend_data.empty:  
        trend_data.to_csv("csgo_trend_data.csv")  
        print("Trend data saved as 'csgo_trend_data.csv'")