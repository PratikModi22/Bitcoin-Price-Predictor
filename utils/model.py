import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_features(df):
    """
    Prepare features for the model
    """
    X = pd.DataFrame({
        'Previous_Close': df['Close'].shift(1),
        'MA7': df['MA7'],
        'MA30': df['MA30'],
        'Daily_Return': df['Daily_Return']
    })
    y = df['Close']
    
    # Remove any NaN values
    X = X.dropna()
    y = y[X.index]
    
    return X, y

def train_model(df):
    """
    Train the linear regression model
    """
    # Prepare features
    X, y = prepare_features(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mse, r2

def make_prediction(model, df):
    """
    Make prediction for the next day
    """
    # Prepare features for the last day
    last_data = pd.DataFrame({
        'Previous_Close': [df['Close'].iloc[-1]],
        'MA7': [df['MA7'].iloc[-1]],
        'MA30': [df['MA30'].iloc[-1]],
        'Daily_Return': [df['Daily_Return'].iloc[-1]]
    })
    
    # Make prediction
    prediction = model.predict(last_data)[0]
    
    return prediction
