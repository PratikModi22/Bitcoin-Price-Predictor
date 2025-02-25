import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from utils.data_handler import fetch_bitcoin_data
from utils.model import train_model, make_prediction

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Bitcoin Price Predictor")
st.markdown("""
This application predicts Bitcoin prices using linear regression based on historical data.
Select a date range to analyze and predict future prices.
""")

# Sidebar controls
st.sidebar.header("Configuration")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

# Fetch data button
if st.sidebar.button("Fetch Data and Predict"):
    try:
        with st.spinner("Fetching Bitcoin data..."):
            df = fetch_bitcoin_data(start_date_input, end_date_input)
            
            if df is not None and not df.empty:
                # Display raw data
                st.subheader("Historical Bitcoin Data")
                st.dataframe(df.tail())
                
                # Create price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='BTC Price'
                ))
                fig.update_layout(
                    title="Bitcoin Price History",
                    yaxis_title="Price (USD)",
                    xaxis_title="Date",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Train model and make predictions
                with st.spinner("Training model and making predictions..."):
                    model, X_test, y_test, y_pred, mse, r2 = train_model(df)
                    next_day_pred = make_prediction(model, df)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Next Day Prediction", f"${next_day_pred:,.2f}")
                    with col2:
                        st.metric("Model R² Score", f"{r2:.4f}")
                    with col3:
                        st.metric("Mean Squared Error", f"{mse:.2f}")
                    
                    # Plot actual vs predicted
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(
                        x=X_test.index,
                        y=y_test,
                        name="Actual Price",
                        line=dict(color="blue")
                    ))
                    pred_fig.add_trace(go.Scatter(
                        x=X_test.index,
                        y=y_pred,
                        name="Predicted Price",
                        line=dict(color="red")
                    ))
                    pred_fig.update_layout(
                        title="Actual vs Predicted Prices",
                        yaxis_title="Price (USD)",
                        xaxis_title="Date",
                        template="plotly_dark"
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("Model Features")
                    st.write("The model uses the following features for prediction:")
                    st.write("- Previous day's closing price")
                    st.write("- 7-day moving average")
                    st.write("- 30-day moving average")
                    st.write("- Daily price change")
                    
            else:
                st.error("No data available for the selected date range.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, yfinance, and scikit-learn")
