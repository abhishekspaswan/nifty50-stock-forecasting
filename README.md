Nifty 50 Stock Forecasting App
Overview
The Nifty 50 Stock Forecasting App is a Streamlit-based web application for analyzing and forecasting stock prices of Nifty 50 companies listed on the NSE. It uses ARIMA, SARIMA, and LSTM models to predict stock prices (1-90 days) and visualizes historical data with candlestick charts and forecasted data with line charts and tables. Model performance is compared using MAPE, MAE, and RMSE metrics. Data is sourced from Yahoo Finance, reflecting prices up to 11:32 AM IST, August 31, 2025. This app is for educational purposes, not financial advice.
Features

Historical Data: View candlestick charts and the last 5 records for selected stocks.
Forecasting: Generate price predictions using ARIMA, SARIMA, and LSTM models.
Model Evaluation: Compare model performance with MAPE, MAE, and RMSE.
Customizable: Select stocks, time periods (1 month to 5 years), and forecast duration (1-90 days).
User-Friendly: Bold, dark-themed interface with Arial font.

Installation

Clone the repository:
git clone https://github.com/abhishekspaswan/nifty50-stock-forecasting.git
cd nifty50-stock-forecasting


Install dependencies:
pip install -r requirements.txt

Dependencies include: streamlit, yfinance, pandas, plotly, statsmodels, scikit-learn, tensorflow, numpy.

Run the app:
streamlit run stock_analysis_app.py

Usage

Open the app in a browser (default: http://localhost:8501).
Select a Nifty 50 stock, time period, and forecast duration.
View historical candlestick charts, forecasted prices, and model performance metrics.

Notes
Data is fetched via the yfinance library from Yahoo Finance.
Forecasts are experimental and should not be used for financial decisions.
The app is optimized for desktop browsers with a dark theme.
Disclaimer
This tool is for educational purposes only. Do not use it for financial advice.
