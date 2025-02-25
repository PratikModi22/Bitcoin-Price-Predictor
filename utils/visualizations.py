import plotly.graph_objects as go
import plotly.express as px
from utils.model import get_feature_importance

def plot_price_history(df):
    """
    Create an interactive plot of Bitcoin price history
    """
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='BTC-USD'
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA5'],
        name='5-day MA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA20'],
        name='20-day MA',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Bitcoin Price History with Moving Averages',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

def plot_prediction(df, predictions, future_predictions):
    """
    Create a plot showing actual vs predicted prices
    """
    fig = go.Figure()
    
    # Plot actual prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions,
        name='Predicted Price',
        line=dict(color='red', dash='dash')
    ))
    
    # Plot future predictions
    future_dates = pd.date_range(
        start=df.index[-1],
        periods=len(future_predictions) + 1,
        closed='right'
    )
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Future Predictions',
        line=dict(color='green', dash='dash')
    ))
    
    fig.update_layout(
        title='Bitcoin Price Predictions',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_white'
    )
    
    return fig

def plot_feature_importance(model):
    """
    Create a bar plot of feature importance
    """
    feature_names = ['Close', 'MA5', 'MA20', 'Volume', 'Volatility']
    importance_dict = get_feature_importance(model, feature_names)
    
    fig = px.bar(
        x=list(importance_dict.keys()),
        y=list(importance_dict.values()),
        title='Feature Importance Analysis'
    )
    
    fig.update_layout(
        xaxis_title='Features',
        yaxis_title='Importance Score',
        template='plotly_white'
    )
    
    return fig
