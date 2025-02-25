import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_bitcoin_data(start_date, end_date):
    """
    Fetch Bitcoin historical data from Yahoo Finance
    """
    try:
        # Add one day to end_date to include it in the range
        end_date_adj = end_date + timedelta(days=1)
        
        # Fetch Bitcoin data
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start=start_date, end=end_date_adj)
        
        if df.empty:
            return None
            
        # Calculate additional features
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Drop any NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")
