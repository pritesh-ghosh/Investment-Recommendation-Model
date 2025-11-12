# Investment Recommendation Model
This project is an end-to-end investment recommendation model built using Python. The model analyzes historical stock data, technical indicators, and news sentiment to recommend an optimal basket of NSE stocks.

## Project Overview
The project is structured as a modular pipeline, with each component responsible for a specific task. This approach ensures the code is clean, manageable, and easy to debug.

## Key Features:

Automated Data Collection: Gathers historical stock data from the National Stock Exchange (NSE).

News Sentiment Analysis: Ingests news articles from RSS feeds, uses Natural Language Processing (NLP) to identify companies, and analyzes the sentiment of the news.

Quantitative Analysis: Calculates technical indicators (e.g., MACD, RSI) for each stock.

Machine Learning Model: Uses a LightGBM classifier to predict future stock price movements based on a combination of quantitative and sentiment data.

Backtesting Simulation: Simulates the model's performance on historical data to evaluate its effectiveness.

Stock Basket Optimization: Recommends a final basket of stocks for investment based on the model's predictions.

## Project Structure
The project is organized into a clear directory structure for optimal workflow.
```
📦 Investment Recommendation Model/
├── 📜 config/
│   ├── settings.ini
│   └── stocks.txt
├── 📦 data/
│   ├── raw/
│   └── processed/
│       └── sentiment/
├── 📜 src/
│   ├── data/
│   │   ├── data_collection.py
│   │   ├── feature_engineering.py
│   │   ├── news_ingestion.py
│   │   └── __init__.py
│   ├── model/
│   │   ├── backtest.py
│   │   ├── optimizer.py
│   │   ├── scoring_model.py
│   │   └── __init__.py
│   └── utils/
│       ├── helpers.py
│       └── __init__.py
├── 📜 main.py
├── 📜 requirements.txt
└── 📜 README.md
```
## Setup and Installation
To run this project, follow these simple steps to set up your environment and install the required libraries.

Step 1: Get the Code
First, copy all the files and folders from this project into a new folder on your computer.

Step 2: Install Libraries
This project uses several Python libraries. You can install all of them at once using the requirements.txt file.

Open your terminal or command prompt.

Navigate to your project's main folder (stock_model).

Run the following command:
```
Bash

pip install -r requirements.txt
```
You also need to download a language model for spaCy and vader. Run this command in your terminal:
```
Bash

python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon
```
Step 3: Configuration
Before running the model, you need to configure your inputs.

config/stocks.txt: Open this file and add the stock tickers you want to analyze, one per line. For Indian stocks, remember to add .NS at the end (e.g., RELIANCE.NS).

## How to Run the Model
The entire project is designed to be run with a single command. The main.py script orchestrates the entire pipeline.

From your terminal, run the following command from the project's root directory:
```
Bash

python main.py
```
## Pipeline and Workflow
The main.py script executes a full pipeline, completing the following steps in order:

Data Collection: Gathers historical OHLCV data for all stocks listed in stocks.txt using the yfinance library. It also cleans up previous runs to avoid redundant data.

News Ingestion: Reads news articles from free RSS feeds, uses spaCy and nltk to identify company mentions, and calculates a sentiment score for each article. This data is saved to data/processed/sentiment.

Feature Engineering: Calculates key technical indicators (MACD, RSI, Bollinger Bands) using the TA-Lib library. This enriched data is saved to data/processed.

Model Training: Combines the quantitative and sentiment data, then trains a LightGBM Classifier to predict whether a stock will move up or down the next day. The trained model is saved as trained_model.joblib.

Backtesting: Simulates the model's performance on historical data to calculate a total return for the investment strategy.

Recommendation: Uses the trained model and the latest data to recommend a final basket of stocks with a predicted probability of upward movement.

This project provides a robust framework for building and evaluating a quantitative investment strategy.
