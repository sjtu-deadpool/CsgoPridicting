from pytrends.request import TrendReq  
import pandas as pd  
import matplotlib.pyplot as plt  
from datetime import datetime  

def fetch_and_save_trend_data(keyword, timeframe="all", region=""):  
    """  
    Fetch Google Trends data for the given keyword and save/plot the trend.  
    
    Args:  
        keyword (str): The topic to track (e.g., "CS:GO").  
        timeframe (str): The timeframe to query (e.g., "today 12-m", "all").  
        region (str): The country-specific interest (default is global).  
    """  
    # Initialize Pytrends  
    pytrends = TrendReq(hl="en-US", tz=360)  
    
    # Build a payload for the keyword  
    pytrends.build_payload([keyword], timeframe=timeframe, geo=region)  
    
    # Fetch interest over time  
    data = pytrends.interest_over_time()  
    
    if data.empty:  
        print("No trend data found for the given keyword.")  
        return  

    # Remove unrelated columns (isPartial)  
    data = data.drop(columns=["isPartial"])  

    # Save the data to CSV  
    filename = f"{keyword}_google_trends_all_time_{datetime.now().strftime('%Y%m%d')}.csv"  
    data.to_csv(filename, index=True)  
    print(f"Trend data saved as {filename}")  

    # Plot the trend  
    plt.figure(figsize=(16, 8))  
    plt.plot(data.index, data[keyword], label=f"Google Trend: {keyword}", color="blue")  
    plt.title(f"Google Trends Interest Over Time for '{keyword}' (All Time)")  
    plt.xlabel("Date")  
    plt.ylabel("Interest Over Time")  
    plt.legend()  
    plt.grid(True)  
    plt.savefig(f"{keyword}_trend_plot_all_time_{datetime.now().strftime('%Y%m%d')}.png")  # Save plot to file  
    plt.show()  


# Example: Fetch and save trend data for the keyword "CS:GO"  
fetch_and_save_trend_data("CS:GO", timeframe="all")  # All available data