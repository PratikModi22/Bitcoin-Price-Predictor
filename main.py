import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.data_handler import fetch_bitcoin_data
from utils.model import train_model, make_prediction
from utils.visualizations import plot_price_history, plot_prediction, plot_feature_importance
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Bitcoin Price Predictor")
st.markdown("""
This application uses machine learning to predict Bitcoin prices based on historical data.
The model analyzes past trends to estimate future values using Linear Regression.
""")

# Sidebar controls
st.sidebar.header("📊 Analysis Parameters")

# Date range selector
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, end_date),
    max_value=end_date
)

# Model parameters
prediction_days = st.sidebar.slider(
    "Number of days for prediction",
    min_value=1,
    max_value=30,
    value=7
)

try:
    # Fetch data
    with st.spinner("Fetching Bitcoin data..."):
        df = fetch_bitcoin_data(date_range[0], date_range[1])
    
    # Display raw data
    st.subheader("Historical Data")
    st.dataframe(df.tail())
    
    # Price history plot
    st.subheader("Bitcoin Price History")
    fig_history = plot_price_history(df)
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Train model and make predictions
    with st.spinner("Training model and making predictions..."):
        model, X_train, y_train, X_test, y_test, predictions, future_predictions = train_model(df, prediction_days)
        
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Data Size", f"{len(X_train)} days")
    with col2:
        st.metric("Testing Data Size", f"{len(X_test)} days")
    with col3:
        st.metric("Model Score", f"{model.score(X_test, y_test):.2%}")
    
    # Prediction visualization
    st.subheader("Price Predictions")
    fig_prediction = plot_prediction(df, predictions, future_predictions)
    st.plotly_chart(fig_prediction, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance Analysis")
    fig_importance = plot_feature_importance(model)
    st.plotly_chart(fig_importance, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try adjusting the date range or refreshing the page.")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This is a prediction model for educational purposes. Do not use it as financial advice.*")
