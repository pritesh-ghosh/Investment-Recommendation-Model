
'''

# src/data/feature_engineering.py

import pandas as pd
import os
import talib

def create_features():
    """
    Reads raw stock data, calculates technical indicators, and saves processed data.
    """
    print("Creating features...")

    # Create the processed data folder if it doesn't exist
    processed_data_folder = "data/processed"
    os.makedirs(processed_data_folder, exist_ok=True)

    # Get a list of all the raw data files
    raw_files = [f for f in os.listdir("data/raw") if f.endswith(".csv")]

    # Loop through each raw data file
    for file in raw_files:
        print(f"  Processing {file}...")
        
        # We specify the correct column names, and tell pandas there's no header.
        df = pd.read_csv(f"data/raw/{file}", 
                         index_col=0, 
                         parse_dates=True, 
                         names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'],
                         skiprows=3)

        # Drop the extra columns that we don't need
        df = df.drop(columns=['Dividends', 'Stock Splits'])

        # Check if the required 'Close' column exists after loading
        if 'Close' not in df.columns:
            print(f"  Skipping {file}: 'Close' column not found. Columns found: {df.columns.tolist()}")
            continue

        # --- Calculate Technical Indicators ---
        # Moving Average Convergence Divergence (MACD)
        macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist

        # Relative Strength Index (RSI)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
        df['Upper_Band'] = upper
        df['Middle_Band'] = middle
        df['Lower_Band'] = lower

        # --- Clean up the data ---
        # The indicators will have missing values at the beginning, so we remove them
        df.dropna(inplace=True)

        # Save the new, processed data to the 'processed' folder
        processed_file_path = os.path.join(processed_data_folder, file)
        df.to_csv(processed_file_path)
    
    print("Feature engineering complete. Processed files are in the 'data/processed' folder.")

# This part allows us to run this script directly to test it
if __name__ == "__main__":
    create_features()
    '''

# src/data/feature_engineering.py

import pandas as pd
import os
import talib
import numpy as np

def create_features():
    """
    Reads raw stock data, calculates technical indicators & engineered features, 
    and saves processed data for modeling.
    """
    print("Creating features...")

    processed_data_folder = "data/processed"
    os.makedirs(processed_data_folder, exist_ok=True)

    raw_files = [f for f in os.listdir("data/raw") if f.endswith(".csv")]

    for file in raw_files:
        print(f"  Processing {file}...")

        try:
            df = pd.read_csv(
                f"data/raw/{file}",
                index_col=0,
                parse_dates=True,
                names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'],
                skiprows=3
            )
        except Exception as e:
            print(f"  Error reading {file}: {e}")
            continue

        df = df.drop(columns=['Dividends', 'Stock Splits'], errors="ignore")

        if 'Close' not in df.columns:
            print(f"  Skipping {file}: 'Close' column not found. Columns found: {df.columns.tolist()}")
            continue

        # --- Technical Indicators ---

        # Moving Averages
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)

        # MACD
        macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist

        # RSI
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
        df['Upper_Band'] = upper
        df['Middle_Band'] = middle
        df['Lower_Band'] = lower

        # Momentum Indicators
        df['Momentum'] = talib.MOM(df['Close'], timeperiod=10)
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        df['Williams_%R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # Volume Indicators
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])

        # Volatility Indicator
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

        # --- Custom Engineered Features ---

        # Daily returns & rolling volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_5d'] = df['Returns'].rolling(window=5).std()
        df['Volatility_10d'] = df['Returns'].rolling(window=10).std()

        # Lag features (yesterday’s close & volume)
        df['Lag1_Close'] = df['Close'].shift(1)
        df['Lag1_Volume'] = df['Volume'].shift(1)

        # Drop NaN rows introduced by indicators
        df.dropna(inplace=True)

        # Save processed file
        processed_file_path = os.path.join(processed_data_folder, file)
        df.to_csv(processed_file_path)

    print("Feature engineering complete. Processed files are in the 'data/processed' folder.")

if __name__ == "__main__":
    create_features()