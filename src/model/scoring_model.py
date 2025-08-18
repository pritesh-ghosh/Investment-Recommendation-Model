# src/model/scoring_model.py
'''
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_scoring_model():
    """
    Loads stock data, merges with sentiment data, and trains a RandomForestClassifier.
    """
    print("Training the scoring model...")
    
    # Step 1: Load all processed stock data files into one big DataFrame
    all_stock_data = []
    processed_files_folder = "data/processed"
    processed_files = [f for f in os.listdir(processed_files_folder) if f.endswith(".csv")]
    
    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_folder, file))
        df['ticker'] = file.replace('.csv', '')
        all_stock_data.append(df)
    
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date']).dt.normalize()

    # Step 2: Load and merge the sentiment data
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
    
    # Merge the two datasets on 'Date' and 'ticker'
    merged_data = pd.merge(all_stock_data, all_sentiment_data, on=['Date', 'ticker'], how='left')
    merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0) # Fill missing sentiment scores with 0
    
    # Step 3: Create our target variable
    merged_data['target'] = (merged_data['Close'].shift(-1) > merged_data['Close']).astype(int)
    
    # Drop the last row of each stock's data as we can't predict its future
    merged_data['is_last_row'] = merged_data.groupby('ticker')['Date'].transform('last') == merged_data['Date']
    merged_data.loc[merged_data['is_last_row'], 'target'] = None
    merged_data.dropna(inplace=True)

    # Step 4: Prepare the data for training
    features = ['Open', 'High', 'Low', 'Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'Sentiment_Score']
    X = merged_data[features]
    y = merged_data['target']

    # Step 5: Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Create and train our model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 7: Test the model's performance
    accuracy = model.score(X_test, y_test)
    print(f"  Model accuracy on test data: {accuracy:.2f}")

    # Step 8: Save the trained model
    model_folder = "src/model"
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(model, os.path.join(model_folder, "trained_model.joblib"))
    print("Model training complete. Model saved as 'trained_model.joblib'.")

if __name__ == "__main__":
    train_scoring_model()
'''

# src/model/scoring_model.py

import pandas as pd
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV

def train_scoring_model():
    """Trains a LightGBM model with hyperparameter tuning and saves it."""
    print("Training the scoring model...")
    
    # Step 1: Load all processed stock data files into one big DataFrame
    all_stock_data = []
    processed_files_folder = "data/processed"
    processed_files = [f for f in os.listdir(processed_files_folder) if f.endswith(".csv")]
    
    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_folder, file))
        df['ticker'] = file.replace('.csv', '')
        all_stock_data.append(df)
        
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date']).dt.normalize()

    # Step 2: Load and merge the sentiment data
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
    
    # Merge the two datasets on 'Date' and 'ticker'
    merged_data = pd.merge(all_stock_data, all_sentiment_data, on=['Date', 'ticker'], how='left')
    merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0)
    
    # Step 3: Create our target variable
    merged_data['target'] = (merged_data['Close'].shift(-1) > merged_data['Close']).astype(int)
    
    # Drop the last row of each stock's data as we can't predict its future
    merged_data['is_last_row'] = merged_data.groupby('ticker')['Date'].transform('last') == merged_data['Date']
    merged_data.loc[merged_data['is_last_row'], 'target'] = None
    merged_data.dropna(inplace=True)

    # Step 4: Prepare the data for training
    features = ['Open', 'High', 'Low', 'Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'Sentiment_Score']
    X = merged_data[features]
    y = merged_data['target']

    # Step 5: Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Create and train our new model with hyperparameter tuning
    print("  Tuning hyperparameters...")
    lgbm_model = lgb.LGBMClassifier(random_state=42)
    
    # Define the parameters we want to test
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1]
    }
    
    # Use GridSearchCV to find the best combination of parameters
    grid_search = GridSearchCV(lgbm_model, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Step 7: Test the model's performance
    accuracy = best_model.score(X_test, y_test)
    print(f"  Model accuracy on test data: {accuracy:.2f}")

    # Step 8: Save the best trained model
    model_folder = "src/model"
    os.makedirs(model_folder, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_folder, "trained_model.joblib"))
    print("Model training complete. Model saved as 'trained_model.joblib'.")

if __name__ == "__main__":
    train_scoring_model()
    '''

import pandas as pd
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime

def train_scoring_model():
    """Trains a LightGBM model with hyperparameter tuning and saves it."""
    print("Training the scoring model...")
    
    # Step 1: Load all processed stock data
    all_stock_data = []
    processed_files_folder = "data/processed"
    processed_files = [f for f in os.listdir(processed_files_folder) if f.endswith(".csv")]
    
    for file in processed_files:
        df = pd.read_csv(os.path.join(processed_files_folder, file))
        df['ticker'] = file.replace('.csv', '')
        all_stock_data.append(df)
        
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date']).dt.normalize()

    # Step 2: Load sentiment data
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
    
    # Step 3: Merge data
    merged_data = pd.merge(all_stock_data, all_sentiment_data, on=['Date', 'ticker'], how='left')
    merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0)
    
    # Shift sentiment to avoid lookahead bias
    merged_data['Sentiment_Score'] = merged_data.groupby('ticker')['Sentiment_Score'].shift(1).fillna(0)
    
    # Step 4: Create target (next day movement)
    merged_data['target'] = (merged_data['Close'].shift(-1) > merged_data['Close']).astype(int)
    merged_data['is_last_row'] = merged_data.groupby('ticker')['Date'].transform('last') == merged_data['Date']
    merged_data.loc[merged_data['is_last_row'], 'target'] = None
    merged_data.dropna(inplace=True)

    # Step 5: Select features automatically
    exclude_cols = ['Date', 'ticker', 'target', 'is_last_row']
    features = [col for col in merged_data.columns if col not in exclude_cols]
    X = merged_data[features]
    y = merged_data['target']

    # Step 6: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # keep time order
    
    # Step 7: LightGBM with tuning
    print("  Tuning hyperparameters...")
    lgbm_model = lgb.LGBMClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63]
    }
    grid_search = GridSearchCV(lgbm_model, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Step 8: Evaluate
    y_pred = best_model.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"  F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred):.2f}")
    
    # Step 9: Save model + metadata
    model_folder = "src/model"
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, "trained_model.joblib")
    metadata = {
        "features": features,
        "trained_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": grid_search.best_params_
    }
    joblib.dump({"model": best_model, "metadata": metadata}, model_path)
    print(f"Model training complete. Saved at {model_path}")

if __name__ == "__main__":
    train_scoring_model()
    '''