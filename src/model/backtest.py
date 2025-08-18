# src/model/backtest.py

import pandas as pd
import os
import joblib

def run_backtest():
    """
    Simulates the investment strategy on historical data.
    """
    print("Starting backtest...")

    # Load the trained model
    model = joblib.load("src/model/trained_model.joblib")
    print("  Trained model loaded.")

    # Load all processed data
    all_stock_data = []
    processed_files_folder = "data/processed"
    processed_files = [f for f in os.listdir(processed_files_folder) if f.endswith(".csv")]
    
    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_folder, file))
        df['ticker'] = file.replace('.csv', '')
        all_stock_data.append(df)
        
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date']).dt.normalize()
    all_stock_data.sort_values(by=['Date', 'ticker'], inplace=True)
    
    # Load sentiment data
    all_sentiment_data = []
    sentiment_folder = "data/processed/sentiment"
    sentiment_files = [f for f in os.listdir(sentiment_folder) if f.endswith("_sentiment.csv")]
    
    for file in sentiment_files:
        df = pd.read_csv(os.path.join(sentiment_folder, file))
        df['ticker'] = file.replace('_sentiment.csv', '')
        all_sentiment_data.append(df)
        
    all_sentiment_data = pd.concat(all_sentiment_data, ignore_index=True) if all_sentiment_data else pd.DataFrame()
    
    if not all_sentiment_data.empty:
        all_sentiment_data['Date'] = pd.to_datetime(all_sentiment_data['Date']).dt.normalize()
    
    # Merge stock and sentiment data
    merged_data = pd.merge(all_stock_data, all_sentiment_data, on=['Date', 'ticker'], how='left')
    merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0)
    
    # Prepare data for prediction
    features = ['Open', 'High', 'Low', 'Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'Sentiment_Score']
    merged_data.dropna(subset=features, inplace=True)

    # Backtest simulation
    print("  Starting simulation...")
    
    portfolio_value = 100000  # Starting with a fictional portfolio of 100,000
    
    # Get unique dates to iterate over
    dates = merged_data['Date'].sort_values().unique()

    for i, date in enumerate(dates):
        if i % 100 == 0:
            print(f"  Simulating day {i+1} of {len(dates)}...")
        
        # Get data for the current and next day
        current_day_data = merged_data[merged_data['Date'] == date].copy()
        next_day_data = merged_data[merged_data['Date'] == (date + pd.Timedelta(days=1))].copy()

        # Check for empty dataframes to prevent errors
        if current_day_data.empty or next_day_data.empty:
            continue

        # Use the model to predict which stocks will go up (class 1)
        predictions = model.predict(current_day_data[features])
        current_day_data['prediction'] = predictions
        
        # Select the stocks predicted to go up
        buy_candidates = current_day_data[current_day_data['prediction'] == 1]
        
        # Check if there are any stocks to buy
        if not buy_candidates.empty:
            # Merge the buy candidates with the next day's data to get their future price
            daily_returns = pd.merge(buy_candidates[['ticker', 'Close']], next_day_data[['ticker', 'Close']], on='ticker', suffixes=('_today', '_next_day'))
            
            # Calculate the daily return of the chosen stocks
            daily_returns['return'] = (daily_returns['Close_next_day'] - daily_returns['Close_today']) / daily_returns['Close_today']
            
            # Calculate the average return and update the portfolio value
            average_daily_return = daily_returns['return'].mean()
            portfolio_value *= (1 + average_daily_return)

    print(f"\nBacktest complete.")
    print(f"Starting portfolio value: ₹100,000")
    print(f"Final portfolio value: ₹{portfolio_value:,.2f}")
    print(f"Total return: {((portfolio_value - 100000) / 100000 * 100):.2f}%")

if __name__ == "__main__":
    run_backtest()
    '''

# src/model/backtest.py
"""
Backtest runner (safe, realistic daily backtest).
- Uses processed CSVs in data/processed/
- Loads trained model from src/model/trained_model.joblib
- Uses next available trading day (no weekend/holiday gap mistakes)
- Equal-weight allocation across selected tickers each day
- Applies transaction costs
- Produces basic performance metrics and saves equity curve to data/backtest_equity.csv
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# CONFIG
PROCESSED_FOLDER = os.path.join("data", "processed")
SENTIMENT_FOLDER = os.path.join(PROCESSED_FOLDER, "sentiment")
MODEL_PATH = os.path.join("src", "model", "trained_model.joblib")
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PER_SIDE = 0.001   # 0.1% per trade side (buy or sell)
PROBABILITY_THRESHOLD = 0.55        # if model supports predict_proba, require this threshold to take a position
EQUITY_OUTPUT = os.path.join("data", "backtest_equity.csv")


def load_processed_data(processed_folder=PROCESSED_FOLDER):
    """Load all per-ticker processed CSVs into a single DataFrame."""
    files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_folder}")

    rows = []
    for f in files:
        df = pd.read_csv(os.path.join(processed_folder, f), parse_dates=["Date"], index_col=None)
        ticker = f.replace(".csv", "")
        if df.empty:
            continue
        df["ticker"] = ticker
        rows.append(df)

    if not rows:
        raise ValueError("No data loaded from processed files.")
    data = pd.concat(rows, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.normalize()
    data.sort_values(["Date", "ticker"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def load_sentiment(sentiment_folder=SENTIMENT_FOLDER):
    """Load all per-ticker daily sentiment files (if present)."""
    if not os.path.exists(sentiment_folder):
        return pd.DataFrame(columns=["Date", "Sentiment_Score", "ticker"])
    files = [f for f in os.listdir(sentiment_folder) if f.endswith("_sentiment.csv")]
    if not files:
        return pd.DataFrame(columns=["Date", "Sentiment_Score", "ticker"])
    rows = []
    for f in files:
        df = pd.read_csv(os.path.join(sentiment_folder, f), parse_dates=["Date"], index_col=None)
        ticker = f.replace("_sentiment.csv", "")
        if df.empty:
            continue
        df["ticker"] = ticker
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["Date", "Sentiment_Score", "ticker"])
    s = pd.concat(rows, ignore_index=True)
    s["Date"] = pd.to_datetime(s["Date"]).dt.normalize()
    return s


def infer_model_and_features(model_path=MODEL_PATH):
    """
    Load model. If joblib stored a dict with model+metadata, extract both.
    Otherwise return the loaded object and None for feature list.
    """
    loaded = joblib.load(model_path)
    model = None
    metadata = None

    # common patterns: saved plain model, or dict {'model': model, 'metadata': {...}}
    if isinstance(loaded, dict):
        model = loaded.get("model", None) or loaded.get("estimator", None) or loaded.get("best_model", None)
        metadata = loaded.get("metadata", None)
        if model is None:
            # maybe user saved {"estimator": GridSearchCV(...)}
            model = loaded
    else:
        model = loaded

    # If metadata contains 'features', use them. Otherwise None.
    feature_list = None
    if metadata and isinstance(metadata, dict):
        feature_list = metadata.get("features", None)

    return model, feature_list


def compute_max_drawdown(equity_series: pd.Series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()


def run_backtest(initial_capital=INITIAL_CAPITAL,
                 transaction_cost_per_side=TRANSACTION_COST_PER_SIDE,
                 prob_threshold=PROBABILITY_THRESHOLD,
                 equity_output=EQUITY_OUTPUT):
    print(f"[{datetime.now()}] Starting backtest...")

    # 1) Load data
    data = load_processed_data()
    sentiment = load_sentiment()
    if not sentiment.empty:
        data = pd.merge(data, sentiment, on=["Date", "ticker"], how="left")
        if "Sentiment_Score" in data.columns:
            data["Sentiment_Score"].fillna(0.0, inplace=True)
    else:
        # ensure column exists for modeling
        if "Sentiment_Score" not in data.columns:
            data["Sentiment_Score"] = 0.0

    # 2) Identify features for model input
    # We'll try to load model metadata (features). If not present, infer numeric features and require Close present.
    model, feature_list = infer_model_and_features()
    if model is None:
        raise FileNotFoundError(f"Could not load model from {MODEL_PATH}")

    # If metadata provided, use it; else infer features from data columns (numeric) but ensure Close/Open present
    if feature_list:
        features = [f for f in feature_list if f in data.columns]
        if not features:
            raise ValueError("Feature list from model metadata doesn't match processed data columns.")
    else:
        # exclude identifier cols
        exclude = {"Date", "ticker"}
        candidate = [c for c in data.columns if c not in exclude and np.issubdtype(data[c].dtype, np.number)]
        # Ensure we include required price columns (Open/High/Low/Close) if available
        features = [c for c in candidate if c.lower() in candidate or True]  # keep numeric cols
        # If too many features, at least ensure Close is included
        if "Close" not in features:
            raise ValueError("Processed data does not contain 'Close' column which is required.")
    print(f"[{datetime.now()}] Using {len(features)} features for prediction.")

    # 3) Build date index (trading calendar)
    dates = sorted(data["Date"].unique())
    if len(dates) < 2:
        raise ValueError("Not enough trading days for backtest.")

    # 4) Prepare structure for daily equity
    portfolio_value = initial_capital
    equity_curve = []
    daily_returns = []

    # 5) Backtest loop (use next available trading day)
    # We'll iterate over date pairs (dates[i], dates[i+1]) to avoid holiday/weekend issues
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]

        current_df = data[data["Date"] == current_date].copy()
        next_df = data[data["Date"] == next_date].copy()

        if current_df.empty or next_df.empty:
            equity_curve.append({"Date": current_date, "Portfolio": portfolio_value})
            daily_returns.append(0.0)
            continue

        # Align features for prediction
        X_today = current_df[features].copy()
        # If model expects features in a specific order, ensure consistent columns
        # (we already filtered features to ones present)
        # Predict probabilities if supported
        use_prob = hasattr(model, "predict_proba")
        try:
            if use_prob:
                probs = model.predict_proba(X_today)[:, 1]
                current_df = current_df.assign(_probability=probs)
                # choose buy candidates by prob threshold
                buy_df = current_df[current_df["_probability"] >= prob_threshold].copy()
            else:
                preds = model.predict(X_today)
                current_df = current_df.assign(_prediction=preds)
                buy_df = current_df[current_df["_prediction"] == 1].copy()
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

        # If no buy candidates, hold cash (portfolio unchanged)
        if buy_df.empty:
            equity_curve.append({"Date": current_date, "Portfolio": portfolio_value})
            daily_returns.append(0.0)
            continue

        # For each buy candidate, compute next-day return using next_df (match on ticker)
        merged = pd.merge(
            buy_df[["ticker", "Close"]],
            next_df[["ticker", "Close"]],
            on="ticker",
            how="inner",
            suffixes=("_today", "_next")
        )

        if merged.empty:
            # no overlapping tickers for next day -> skip
            equity_curve.append({"Date": current_date, "Portfolio": portfolio_value})
            daily_returns.append(0.0)
            continue

        # compute raw returns
        merged["raw_return"] = (merged["Close_next"] - merged["Close_today"]) / merged["Close_today"]

        # account for transaction costs: when opening and closing a position we pay cost per side
        # cost as fraction of notional. For equal-weight allocation across N positions:
        n_positions = len(merged)
        weight = 1.0 / n_positions

        # Apply transaction costs: subtract 2 * cost per side from each position's return (approx)
        merged["net_return"] = merged["raw_return"] - 2 * transaction_cost_per_side

        # portfolio daily return = weighted avg of net returns
        portfolio_daily_return = (merged["net_return"] * weight).sum()

        # Update portfolio value by compounding
        new_portfolio = portfolio_value * (1.0 + portfolio_daily_return)

        # Record daily return and equity
        equity_curve.append({"Date": current_date, "Portfolio": portfolio_value})
        daily_returns.append(portfolio_daily_return)

        # Move portfolio forward
        portfolio_value = new_portfolio

    # Append last day equity
    equity_curve.append({"Date": dates[-1], "Portfolio": portfolio_value})

    # Build equity DataFrame
    equity_df = pd.DataFrame(equity_curve)
    equity_df["Date"] = pd.to_datetime(equity_df["Date"])
    equity_df.sort_values("Date", inplace=True)
    equity_df.reset_index(drop=True, inplace=True)

    # compute performance metrics
    equity_df["Pct_Return"] = equity_df["Portfolio"].pct_change().fillna(0)
    cumulative_return = equity_df["Portfolio"].iloc[-1] / equity_df["Portfolio"].iloc[0] - 1.0
    num_days = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days
    trading_days = len(equity_df) - 1
    # annualize using trading days
    if trading_days > 0:
        CAGR = (equity_df["Portfolio"].iloc[-1] / equity_df["Portfolio"].iloc[0]) ** (252.0 / trading_days) - 1.0
        annual_vol = equity_df["Pct_Return"].std() * np.sqrt(252)
    else:
        CAGR = 0.0
        annual_vol = 0.0

    sharpe = (CAGR / annual_vol) if annual_vol != 0 else 0.0
    max_dd = compute_max_drawdown(equity_df["Portfolio"])

    results = {
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Portfolio": round(portfolio_value, 2),
        "Cumulative_Return_%": round(cumulative_return * 100, 2),
        "CAGR_%": round(CAGR * 100, 2),
        "Annualized_Volatility_%": round(annual_vol * 100, 2),
        "Sharpe_Ratio": round(sharpe, 3),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Trading_Days": trading_days,
    }

    # Save equity curve
    os.makedirs(os.path.dirname(equity_output), exist_ok=True)
    equity_df.to_csv(equity_output, index=False)
    print(f"[{datetime.now()}] Backtest complete. Equity curve saved to {equity_output}")

    # Basic sanity check
    if results["Cumulative_Return_%"] > 1000:
        print("⚠️ Warning: cumulative return > 1000%. Check for leakage or unrealistic assumptions.")

    return results, equity_df


if __name__ == "__main__":
    res, eq = run_backtest()
    print("\nBacktest Summary:")
    for k, v in res.items():
        print(f"  {k}: {v}")
'''