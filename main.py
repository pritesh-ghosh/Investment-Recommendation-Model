# main.py

import sys
import os

# Add the 'src' directory to the system path so you can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from our scripts
from data.data_collection import collect_stock_data
from data.news_ingestion import process_news
from data.feature_engineering import create_features
from model.scoring_model import train_scoring_model
from model.backtest import run_backtest
from model.optimizer import recommend_stocks
from utils.helpers import clean_directory # Import the new function

def run_full_pipeline():
    """
    Executes the full investment model pipeline in the correct order.
    """
    print("--- Starting the full investment model pipeline ---")

    # --- NEW CODE: Clean up old data ---
    print("\nCleaning up old data...")
    clean_directory("data/raw")
    clean_directory("data/processed")
    clean_directory("data/processed/sentiment")
    # --- END OF NEW CODE ---

    # Phase 1: Data Collection
    print("\nPhase 1: Collecting data...")
    collect_stock_data()

    # Phase 2: News Ingestion & Sentiment Analysis
    print("\nPhase 2: Processing news for sentiment...")
    process_news()

    # Phase 3: Feature Engineering
    print("\nPhase 3: Creating features and indicators...")
    create_features()

    # Phase 4: Model Training
    print("\nPhase 4: Training the scoring model...")
    train_scoring_model()

    # Phase 5: Backtesting
    print("\nPhase 5: Running the backtest simulation...")
    run_backtest()

    # Phase 6: Generating Recommendations
    print("\nPhase 6: Generating a final stock basket recommendation...")
    recommend_stocks(5) # Recommends a basket of 5 stocks

    print("\n--- Pipeline complete! ---")

if __name__ == "__main__":
    run_full_pipeline()