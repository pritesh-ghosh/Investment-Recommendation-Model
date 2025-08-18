# src/model/optimizer.py

import pandas as pd
import os
import joblib

def recommend_stocks(n):
    """
    Loads the trained model and recommends a basket of n stocks.
    """
    print("Generating stock recommendations...")

    # Load the trained model
    try:
        model = joblib.load("src/model/trained_model.joblib")
        print("  Trained model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model not found. Please run scoring_model.py first.")
        return

    # Load the most recent data for all stocks
    all_data = []
    processed_files_folder = "data/processed"
    processed_files = [f for f in os.listdir(processed_files_folder) if f.endswith(".csv")]
    
    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_folder, file))
        
        # --- ADDED CODE TO HANDLE EMPTY DATAFRAMES ---
        if not df.empty:
            # Get the very last row (most recent day's data)
            most_recent_data = df.iloc[-1].to_dict()
            most_recent_data['ticker'] = file.replace('.csv', '')
            all_data.append(most_recent_data)
        # --- END OF ADDED CODE ---

    df_recent = pd.DataFrame(all_data)
    
    # Load sentiment data
    all_sentiment_data = []
    sentiment_folder = "data/processed/sentiment"
    sentiment_files = [f for f in os.listdir(sentiment_folder) if f.endswith("_sentiment.csv")]
    
    for file in sentiment_files:
        df = pd.read_csv(os.path.join(sentiment_folder, file))
        df['ticker'] = file.replace('_sentiment.csv', '')
        all_sentiment_data.append(df)
        
    df_recent_sentiment = pd.concat(all_sentiment_data, ignore_index=True) if all_sentiment_data else pd.DataFrame()
    df_recent_sentiment['Date'] = pd.to_datetime(df_recent_sentiment['Date']).dt.normalize()
    
    # Merge sentiment into the recent data
    df_recent = pd.merge(df_recent, df_recent_sentiment, on='ticker', how='left')
    df_recent['Sentiment_Score'] = df_recent['Sentiment_Score'].fillna(0)
    
    # Remove duplicates from recent data
    df_recent.drop_duplicates(subset=['ticker'], keep='last', inplace=True)
    
    # Prepare the data for prediction
    features = ['Open', 'High', 'Low', 'Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'Sentiment_Score']
    df_recent.dropna(subset=features, inplace=True)
    
    if df_recent.empty:
        print("Error: No data available to make a prediction.")
        return

    # Use the model to predict the probability of each stock going up
    df_recent['prediction'] = model.predict_proba(df_recent[features])[:, 1]
    
    # Sort the stocks by their predicted probability
    recommendations = df_recent.sort_values(by='prediction', ascending=False)
    
    # Select the top N stocks
    top_n_recommendations = recommendations.head(n)
    
    print("\nRecommended Stock Basket:")
    print("-" * 30)
    for index, row in top_n_recommendations.iterrows():
        print(f"{row['ticker']}: Predicted Upward Probability = {row['prediction']:.2f}")

    return top_n_recommendations

if __name__ == "__main__":
    # We want a basket of 5 stocks
    recommend_stocks(5)