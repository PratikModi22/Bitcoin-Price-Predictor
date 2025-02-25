import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils.data_handler import prepare_data

def train_model(df, prediction_days):
    """
    Train the linear regression model and make predictions
    """
    # Prepare the data
    X, y = prepare_data(df, prediction_days)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    predictions = model.predict(X_test)
    
    # Prepare future prediction data
    last_data = df[['Close', 'MA5', 'MA20', 'Volume', 'Volatility']].tail(prediction_days)
    future_predictions = model.predict(last_data)
    
    return model, X_train, y_train, X_test, y_test, predictions, future_predictions

def get_feature_importance(model, feature_names):
    """
    Get feature importance scores from the model
    """
    importance = abs(model.coef_)
    importance_dict = dict(zip(feature_names, importance))
    return importance_dict
