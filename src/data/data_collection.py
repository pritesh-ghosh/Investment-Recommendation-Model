# src/data/data_collection.py

import yfinance as yf
import pandas as pd
import os

def collect_stock_data():
    """
    Downloads historical stock data for tickers listed in stocks.txt.
    """
    print("Collecting stock data...")
    
    # Step 1: Read the list of tickers from our stocks.txt file
    try:
        with open("config/stocks.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()] 
        print(f"Found {len(tickers)} tickers in stocks.txt.")
    except FileNotFoundError:
        print("Error: The 'stocks.txt' file was not found in the 'config' folder.")
        return

    # Step 2: Make sure the raw data folder exists
    raw_data_folder = "data/raw"
    os.makedirs(raw_data_folder, exist_ok=True)

    # Step 3: Loop through each ticker and download its data
    for ticker in tickers:
        print(f"  Downloading data for {ticker}...")
        try:
            # yf.download gets the data. 'period=5y' means we want 5 years of data.
            data = yf.download(ticker, period="5y")
            
            # Check if we got any data back
            if not data.empty:
                # Save the data to a CSV file in the 'data/raw' folder
                file_path = os.path.join(raw_data_folder, f"{ticker}.csv")
                data.to_csv(file_path)
                print(f"  Successfully saved data to {file_path}")
            else:
                print(f"  No data found for {ticker}.")
                
        except Exception as e:
            print(f"  Failed to download {ticker}: {e}")

    print("Data collection complete.")

# This part allows us to run this script directly to test it
if __name__ == "__main__":
    collect_stock_data()