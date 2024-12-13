import requests  
import pandas as pd  
from datetime import datetime  
import matplotlib.pyplot as plt  

class CSGOHistoricalData:  
    def __init__(self):  
        self.base_url = "https://steamcharts.com/app/730/chart-data.json"  
        self.headers = {  
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  
        }  
        self.smoothing_window = 7  # Moving average window (e.g., 7 days)  

    def get_historical_data(self):  
        """  
        Fetch CS:GO historical player data.  
        """  
        try:  
            response = requests.get(self.base_url, headers=self.headers)  
            if response.status_code == 200:  
                data = response.json()  
                return data  
            else:  
                print(f"Error: Failed to fetch data. Status Code: {response.status_code}")  
                return None  
        except Exception as e:  
            print(f"Exception occurred while fetching data: {e}")  
            return None  

    def process_data(self, data):  
        """  
        Process raw API data into a DataFrame of daily peak player counts.  
        """  
        # Create a DataFrame from raw data  
        df = pd.DataFrame(data, columns=['timestamp', 'avg_players'])  
        
        # Convert timestamp from milliseconds to datetime  
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  
        
        # Extract the date part (YYYY-MM-DD)  
        df['date'] = df['timestamp'].dt.date  
        
        # Group by date and compute the daily peak player count  
        peak_data = df.groupby('date').agg(  
            peak_players=('avg_players', 'max')  # Daily peak player count  
        ).reset_index()  

        return peak_data  

    def smooth_data(self, df):  
        """  
        Smooth the peak players data using a moving average method.  
        """  
        df['smoothed_peak_players'] = df['peak_players'].rolling(window=self.smoothing_window, min_periods=1).mean()  
        return df  

    def save_data(self, df: pd.DataFrame, filename: str = None):  
        """  
        Save the smoothed peak player data as both CSV and JSON.  
        """  
        if filename is None:  
            filename = f"csgo_peak_players_smoothed_moving_average_{datetime.now().strftime('%Y%m%d')}"  

        # Save as CSV  
        df.to_csv(f"{filename}.csv", index=False)  
        
        # Save as JSON  
        df[['date', 'smoothed_peak_players']].to_json(f"{filename}.json", orient='records', date_format='iso')  
        
        print(f"Data successfully saved as: {filename}.csv and {filename}.json")  

    def plot_data(self, df: pd.DataFrame):  
        """  
        Plot the smoothed peak player trend alongside original data.  
        """  
        plt.figure(figsize=(12, 6))  
        plt.plot(df['date'], df['peak_players'], label='Original Peak Players', color='red', alpha=0.5)  
        plt.plot(df['date'], df['smoothed_peak_players'], label='Smoothed Peak Players (Moving Average)', color='blue')  
        plt.xlabel('Date')  
        plt.ylabel('Peak Players')  
        plt.title('CS:GO Daily Peak Player Data (Smoothed)')  
        plt.legend()  
        plt.grid(True)  
        plt.show()  

    def run(self):  
        """  
        Fetch, process, smooth, save, and plot the peak player data.  
        """  
        print("Starting processing of CS:GO player data...")  

        # Fetch raw data from the API  
        raw_data = self.get_historical_data()  
        if raw_data is None:  
            print("Error: Unable to fetch data.")  
            return  

        # Process data to extract peak players  
        peak_data = self.process_data(raw_data)  

        # Smooth the data using a moving average  
        smoothed_data = self.smooth_data(peak_data)  

        # Save the smoothed peak player data  
        self.save_data(smoothed_data)  

        # Plot the smoothed peak player trend  
        self.plot_data(smoothed_data)  

        print("\nProcessing complete. Smoothed peak player data has been saved.")  

def main():  
    collector = CSGOHistoricalData()  
    collector.run()  

if __name__ == "__main__":  
    main()