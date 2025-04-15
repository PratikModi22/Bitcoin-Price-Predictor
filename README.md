# Bitcoin Price Predictor

This project is a machine learning-powered web app that predicts Bitcoin prices using historical data. Built with Python, Streamlit, and scikit-learn, it offers interactive charts and future price forecasting.

## 🚀 Features

- Fetches real-time Bitcoin data from Yahoo Finance
- Calculates technical indicators:
  - 7-day and 30-day Moving Averages
  - Daily Returns
- Trains a Linear Regression model to forecast prices
- Predicts:
  - **Next Day Bitcoin Price**
  - **30-Day Future Prices**
- Visualizes data using interactive Plotly charts
- Displays evaluation metrics (R² Score, Mean Squared Error)

## 🛠️ Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- scikit-learn
- plotly

## 📁 Project Structure

```
Bitcoin-Price-Predictor/
│
├── app.py                  # Streamlit app
├── requirements.txt        # Required Python packages
│
├── utils/
│   ├── data_handler.py     # Data fetching and preprocessing
│   └── model.py            # Model training and prediction logic
```

## 🧪 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/PratikModi22/Bitcoin-Price-Predictor.git
cd Bitcoin-Price-Predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

## 📊 Model Metrics (Sample Output)

- **Next Day Prediction:** $86,931.94  
- **Model R² Score:** 0.9951  
- **Mean Squared Error:** 178,431.78  

## 🔗 GitHub Repository

[https://github.com/PratikModi22/Bitcoin-Price-Predictor](https://github.com/PratikModi22/Bitcoin-Price-Predictor)
```

---

Let me know if you want to add deployment instructions for Streamlit Cloud or badges (like build passing, license, etc.).
