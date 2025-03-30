import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        # Check if data is empty
        if data.empty:
            st.error("No data fetched. Please check the tickers and date range.")
            return None
        return data['Adj Close']  # Return only the 'Adj Close' prices
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to calculate performance metrics
def calculate_metrics(daily_returns, weights):
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)  # Annualized Sharpe Ratio
    max_drawdown = (portfolio_returns.cumsum().min() - portfolio_returns.cumsum().max()) / portfolio_returns.cumsum().max()
    volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized Volatility
    return sharpe_ratio, max_drawdown, volatility

# Streamlit app
st.title("Stock Portfolio Analyzer")

# User input
st.sidebar.header("User  Input")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL, MSFT, GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Process user input
tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]

# Fetch stock data
if st.sidebar.button("Analyze"):
    stock_data = fetch_stock_data(tickers_list, start_date, end_date)

    if stock_data is not None:
        # Display stock data
        st.subheader("Stock Prices")
        st.line_chart(stock_data)

        # Calculate daily returns
        daily_returns = stock_data.pct_change()
        st.subheader("Daily Returns")
        st.line_chart(daily_returns)

        # Portfolio performance
        weights = st.sidebar.text_input("Enter Portfolio Weights (comma-separated)", "0.33, 0.33, 0.34")
        weights_list = [float(weight) for weight in weights.split(",")]

        # Validate weights
        if sum(weights_list) != 1.0:
            st.error("Weights must sum to 1. Please adjust your weights.")
        else:
            # Calculate portfolio returns and metrics
            portfolio_returns = (daily_returns * weights_list).sum(axis=1)
            sharpe_ratio, max_drawdown, volatility = calculate_metrics(daily_returns, weights_list)

            # Display performance metrics
            st.subheader("Performance Metrics")
            st.write(f"*Sharpe Ratio:* {sharpe_ratio:.2f}")
            st.write(f"*Maximum Drawdown:* {max_drawdown:.2%}")
            st.write(f"*Annualized Volatility:* {volatility:.2%}")

            # Cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            st.subheader("Cumulative Portfolio Returns")
            st.line_chart(cumulative_returns)

            # Display stock data table
            st.subheader("Stock Data Table")
            st.dataframe(stock_data)

            # Average daily returns
            avg_daily_returns = daily_returns.mean()
            st.subheader("Average Daily Returns")
            st.bar_chart(avg_daily_returns)

            # Pie chart for portfolio weights
            st.subheader("Portfolio Weights")
            fig, ax = plt.subplots()
            ax.pie(weights_list, labels=tickers_list, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

# Footer
st.sidebar.markdown("### About")
st.sidebar.markdown("This app allows you to analyze your stock portfolio by visualizing stock prices, daily returns, cumulative returns, average daily returns, and portfolio weights.")
