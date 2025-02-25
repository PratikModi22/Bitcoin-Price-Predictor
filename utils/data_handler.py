import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_bitcoin_data(start_date, end_date):
    """
    Fetch Bitcoin historical data from Yahoo Finance
    """
    try:
        # Fetch BTC-USD data
        btc = yf.download('BTC-USD', start=start_date, end=end_date)
        
        # Basic data cleaning
        btc = btc.dropna()
        
        # Add technical indicators
        btc['MA5'] = btc['Close'].rolling(window=5).mean()
        btc['MA20'] = btc['Close'].rolling(window=20).mean()
        btc['Daily_Return'] = btc['Close'].pct_change()
        btc['Volatility'] = btc['Daily_Return'].rolling(window=20).std()
        
        # Clean up any remaining NaN values
        btc = btc.dropna()
        
        return btc
    
    except Exception as e:
        raise Exception(f"Error fetching Bitcoin data: {str(e)}")

def prepare_data(df, prediction_days=7):
    """
    Prepare data for model training
    """
    # Create features
    df['Target'] = df['Close'].shift(-1)
    
    # Create feature matrix
    features = ['Close', 'MA5', 'MA20', 'Volume', 'Volatility']
    X = df[features].iloc[:-prediction_days]
    y = df['Target'].iloc[:-prediction_days]
    
    # Remove any remaining NaN values
    X = X.dropna()
    y = y.dropna()
    
    return X, y
