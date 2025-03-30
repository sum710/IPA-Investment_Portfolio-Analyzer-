import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)

        # Debugging step: Print columns to verify what data is returned
        st.write("Fetched Data Preview:", data.head())

        # Handle MultiIndex if multiple tickers are provided
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs('Adj Close', axis=1, level=1, drop_level=True)
        elif 'Adj Close' in data:
            data = data[['Adj Close']]
        elif 'Close' in data:
            data = data[['Close']]
        else:
            st.error("Stock data is missing. Please check the ticker symbols.")
            return None

        return data
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
st.title("📈 Stock Portfolio Analyzer")

# User input
st.sidebar.header("User Input")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL, MSFT, GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Process user input
tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]

# Fetch stock data
if st.sidebar.button("Analyze"):
    stock_data = fetch_stock_data(tickers_list, start_date, end_date)

    if stock_data is not None and not stock_data.empty:
        # Ensure the index is datetime
        stock_data.index = pd.to_datetime(stock_data.index)

        # Display stock data
        st.subheader("📊 Stock Prices Over Time")
        st.line_chart(stock_data)

        # Calculate daily returns
        daily_returns = stock_data.pct_change().dropna()
        st.subheader("📉 Daily Returns")
        st.line_chart(daily_returns)

        # Portfolio performance
        weights = st.sidebar.text_input("Enter Portfolio Weights (comma-separated)", "0.33, 0.33, 0.34")
        try:
            weights_list = [float(weight) for weight in weights.split(",")]
            if len(weights_list) != len(tickers_list) or sum(weights_list) != 1.0:
                st.error("⚠ Weights must match the number of stocks and sum to 1.")
            else:
                # Calculate portfolio returns and metrics
                portfolio_returns = (daily_returns * weights_list).sum(axis=1)
                sharpe_ratio, max_drawdown, volatility = calculate_metrics(daily_returns, weights_list)

                # Display performance metrics
                st.subheader("📊 Performance Metrics")
                st.write(f"📌 **Sharpe Ratio:** {sharpe_ratio:.2f}")
                st.write(f"📌 **Maximum Drawdown:** {max_drawdown:.2%}")
                st.write(f"📌 **Annualized Volatility:** {volatility:.2%}")

                # Cumulative returns
                cumulative_returns = (1 + portfolio_returns).cumprod()
                st.subheader("📈 Cumulative Portfolio Returns")
                st.line_chart(cumulative_returns)

                # Moving averages (50-day & 200-day)
                st.subheader("📊 50-Day & 200-Day Moving Averages")
                ma_50 = stock_data.rolling(window=50).mean()
                ma_200 = stock_data.rolling(window=200).mean()
                plt.figure(figsize=(10, 5))
                plt.plot(stock_data, label="Stock Prices", alpha=0.7)
                plt.plot(ma_50, label="50-Day MA", linestyle="dashed")
                plt.plot(ma_200, label="200-Day MA", linestyle="dashed")
                plt.legend()
                st.pyplot(plt)

                # Risk vs Return scatter plot
                st.subheader("📊 Risk vs Return Scatter Plot")
                avg_daily_returns = daily_returns.mean()
                std_deviation = daily_returns.std()
                plt.figure(figsize=(6, 4))
                plt.scatter(std_deviation, avg_daily_returns, c="blue", label="Stocks")
                plt.xlabel("Risk (Standard Deviation)")
                plt.ylabel("Return (Mean Daily Return)")
                plt.title("Risk vs Return")
                for i, txt in enumerate(tickers_list):
                    plt.annotate(txt, (std_deviation[i], avg_daily_returns[i]))
                st.pyplot(plt)

                # Display stock data table
                st.subheader("📄 Stock Data Table")
                st.dataframe(stock_data)

                # Pie chart for portfolio weights
                st.subheader("📊 Portfolio Weights Distribution")
                fig, ax = plt.subplots()
                ax.pie(weights_list, labels=tickers_list, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

        except ValueError:
            st.error("⚠ Invalid weight input. Please enter numeric values separated by commas.")

# Footer
st.sidebar.markdown("### ℹ About")
st.sidebar.markdown("This app analyzes your stock portfolio by visualizing stock prices, returns, risk-return tradeoff, and performance metrics.")
