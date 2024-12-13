import requests  
import pandas as pd  
from datetime import datetime, timedelta  
import json  
import matplotlib.pyplot as plt  

class CSGOHistoricalData:  
    def __init__(self):  
        self.base_url = "https://steamcharts.com/app/730/chart-data.json"  
        self.headers = {  
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  
        }  
        
    def get_historical_data(self):  
        """  
        Fetch CS:GO historical player data.  
        Returns: A list containing timestamps and the number of average players.  
        """  
        try:  
            response = requests.get(self.base_url, headers=self.headers)  
            if response.status_code == 200:  
                data = response.json()  
                return data  
            else:  
                print(f"Request failed with status code: {response.status_code}")  
                return None  
        except Exception as e:  
            print(f"Error fetching data: {e}")  
            return None  

    def process_data(self, data):  
        """  
        Process raw data into daily intervals as a pandas DataFrame.  
        """  
        # Create a DataFrame from the raw data with columns: 'timestamp' and 'avg_players'.  
        df = pd.DataFrame(data, columns=['timestamp', 'avg_players'])  
        
        # Convert the timestamp (in milliseconds) to datetime.  
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  
        
        # Add a 'date' column that only captures the YYYY-MM-DD part.  
        df['date'] = df['timestamp'].dt.date  
        
        # Aggregate data by day: calculate the daily average and the daily peak.  
        daily_data = df.groupby('date').agg(  
            avg_players=('avg_players', 'mean'),  # Daily average players  
            peak_players=('avg_players', 'max')  # Daily peak players  
        ).reset_index()  

        return daily_data  

    def save_data(self, df: pd.DataFrame, filename: str = None):  
        """  
        Save the aggregated daily data as both CSV and JSON.  
        """  
        if filename is None:  
            filename = f"csgo_daily_data_{datetime.now().strftime('%Y%m%d')}"  
        
        # Save as CSV  
        df.to_csv(f"{filename}.csv", index=False)  
        
        # Save as JSON  
        df.to_json(f"{filename}.json", orient='records', date_format='iso')  
        
        print(f"Data successfully saved to: {filename}.csv and {filename}.json")  

    def plot_data(self, df: pd.DataFrame):  
        """  
        Plot trends of daily player data.  
        """  
        plt.figure(figsize=(12, 6))  
        plt.plot(df['date'], df['avg_players'], label='Average Players', color='blue')  
        plt.plot(df['date'], df['peak_players'], label='Peak Players', linestyle='--', color='red')  
        plt.xlabel('Date')  
        plt.ylabel('Number of Players')  
        plt.title('CS:GO Daily Player Data')  
        plt.legend()  
        plt.grid(True)  
        plt.show()  

    def run(self):  
        """  
        Execute the full workflow: data fetching, processing, and saving.  
        """  
        print("Starting to process CS:GO player data...")  

        # Fetch raw data from the API.  
        raw_data = self.get_historical_data()  
        if raw_data is None:  
            print("Failed to fetch data.")  
            return  

        # Process the data into daily intervals.  
        daily_data = self.process_data(raw_data)  

        # Save the daily-aggregated data:  
        self.save_data(daily_data)  

        # Print statistics:  
        print(f"\nProcessing complete! Data summary:")  
        print(f"Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")  
        print(f"Total days: {len(daily_data)}")  
        print(f"Daily Average Players (overall mean): {daily_data['avg_players'].mean():.2f}")  
        print(f"Highest Peak Players (daily max): {daily_data['peak_players'].max()}")  

        # Plot the data:  
        self.plot_data(daily_data)  

def main():  
    collector = CSGOHistoricalData()  
    collector.run()  

if __name__ == "__main__":  
    main()