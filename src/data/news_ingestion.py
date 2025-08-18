# src/data/news_ingestion.py

import feedparser
import spacy
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tldextract
import re

# Load the spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

def get_sentiment_from_text(text):
    """
    Analyzes the sentiment of a given text and returns a compound score.
    """
    return sid.polarity_scores(text)['compound']

def get_aliases_from_tickers():
    """
    Reads the tickers from stocks.txt and creates a dictionary of company aliases.
    """
    try:
        with open("config/stocks.txt", "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: The 'stocks.txt' file was not found.")
        return {}

    aliases = {}
    for ticker in tickers:
        # We use the part of the ticker before '.NS' as the company name
        company_name = ticker.split('.')[0]
        aliases[company_name] = ticker
    return aliases

def process_news():
    """
    Pulls news from RSS feeds, finds companies, and analyzes sentiment.
    """
    print("Processing news for sentiment...")

    # Your list of RSS feed URLs
    rss_urls = [
        "https://news.google.com/rss/search?q=NSE+stocks+india&hl=en-IN&gl=IN&ceid=IN:en",
        "https://www.moneycontrol.com/rss/marketreports.xml"
    ]

    all_articles = []
    
    for url in rss_urls:
        print(f"  Fetching news from {url}...")
        feed = feedparser.parse(url)
        for entry in feed.entries:
            # Extract relevant information
            title = entry.title if 'title' in entry else ''
            summary = entry.summary if 'summary' in entry else ''
            link = entry.link if 'link' in entry else ''
            published = entry.published if 'published' in entry else ''
            
            # Use tldextract to get the domain for deduplication
            domain = tldextract.extract(link).domain if link else ''
            
            # Combine title and summary for analysis
            full_text = f"{title}. {summary}"
            
            all_articles.append({
                'title': title,
                'summary': summary,
                'link': link,
                'domain': domain,
                'published': published,
                'full_text': full_text
            })

    # Deduplicate articles based on link
    df_articles = pd.DataFrame(all_articles).drop_duplicates(subset=['link'])
    print(f"  Found {len(df_articles)} unique articles.")
    
    # --- Corrected code to handle dates with a more robust method ---
    df_articles['published'] = pd.to_datetime(df_articles['published'], errors='coerce')
    df_articles.dropna(subset=['published'], inplace=True) # Drop any rows where the date couldn't be parsed
    df_articles['published'] = df_articles['published'].dt.tz_localize(None) # Make all dates timezone-naive
    # --- End of corrected code ---
    
    # Get aliases for fuzzy matching
    aliases = get_aliases_from_tickers()
    sentiment_by_company = {ticker: [] for ticker in aliases.values()}
    
    for index, row in df_articles.iterrows():
        doc = nlp(row['full_text'])
        
        found_companies = set()
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                # Fuzzy match the organization name to our list of companies
                for name, ticker in aliases.items():
                    # We use a regex search for the company name
                    if re.search(r'\b' + re.escape(name) + r'\b', ent.text, re.IGNORECASE):
                        found_companies.add(ticker)
        
        if found_companies:
            sentiment_score = get_sentiment_from_text(row['full_text'])
            for ticker in found_companies:
                sentiment_by_company[ticker].append({
                    'Date': row['published'].normalize(), # Use the now tz-naive date
                    'Sentiment_Score': sentiment_score
                })

    # Save the sentiment data to files
    sentiment_folder = "data/processed/sentiment"
    os.makedirs(sentiment_folder, exist_ok=True)
    
    for ticker, sentiments in sentiment_by_company.items():
        if sentiments:
            df = pd.DataFrame(sentiments)
            # Group by date and take the average sentiment score
            daily_sentiment = df.groupby('Date')['Sentiment_Score'].mean().reset_index()
            daily_sentiment.to_csv(os.path.join(sentiment_folder, f"{ticker}_sentiment.csv"), index=False)
            print(f"  Saved sentiment data for {ticker}.")
        else:
            print(f"  No articles found for {ticker}.")

    print("News ingestion and sentiment analysis complete.")

if __name__ == "__main__":
    process_news()