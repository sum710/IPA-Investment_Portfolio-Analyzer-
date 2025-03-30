import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        st.write("Fetched Data Preview:", data.head())  # Debugging step
        
        # Handle MultiIndex if multiple tickers are provided
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(1):  # Use 'Close' instead of 'Adj Close'
                data = data.xs('Close', axis=1, level=1, drop_level=True)
            else:
                st.error("Yahoo Finance did not return 'Close'. Check ticker symbols.")
                return None
        elif 'Close' in data.columns:
            data = data[['Close']]
        else:
            st.error("No suitable stock data found. Check ticker symbols and dates.")
            return None
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

            

# Function to calculate portfolio performance
def calculate_metrics(daily_returns, weights):
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = (portfolio_returns.cumsum().min() - portfolio_returns.cumsum().max()) / portfolio_returns.cumsum().max()
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    return sharpe_ratio, max_drawdown, volatility

# Streamlit UI
st.title("📈 Investment Portfolio Analyzer")

st.sidebar.header("User Input")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL, MSFT, GOOGL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

tickers_list = [ticker.strip().upper() for ticker in tickers.split(",")]

if st.sidebar.button("Analyze"):
    stock_data = fetch_stock_data(tickers_list, start_date, end_date)
    if stock_data is not None and not stock_data.empty:
        stock_data.index = pd.to_datetime(stock_data.index)
        
        st.subheader("📊 Stock Prices Over Time")
        st.line_chart(stock_data)
        
        daily_returns = stock_data.pct_change().dropna()
        st.subheader("📉 Daily Returns")
        st.line_chart(daily_returns)
        
        weights = st.sidebar.text_input("Enter Portfolio Weights (comma-separated)", "0.33, 0.33, 0.34")
        try:
            weights_list = [float(weight) for weight in weights.split(",")]
            if len(weights_list) != len(tickers_list) or sum(weights_list) != 1.0:
                st.error("⚠ Weights must match stock count & sum to 1.")
            else:
                portfolio_returns = (daily_returns * weights_list).sum(axis=1)
                sharpe_ratio, max_drawdown, volatility = calculate_metrics(daily_returns, weights_list)
                
                st.subheader("📊 Performance Metrics")
                st.write(f"📌 **Sharpe Ratio:** {sharpe_ratio:.2f}")
                st.write(f"📌 **Max Drawdown:** {max_drawdown:.2%}")
                st.write(f"📌 **Annualized Volatility:** {volatility:.2%}")
                
                cumulative_returns = (1 + portfolio_returns).cumprod()
                st.subheader("📈 Cumulative Portfolio Returns")
                st.line_chart(cumulative_returns)
                
                st.subheader("📊 50-Day & 200-Day Moving Averages")
                ma_50 = stock_data.rolling(window=50).mean()
                ma_200 = stock_data.rolling(window=200).mean()
                plt.figure(figsize=(10, 5))
                plt.plot(stock_data, label="Stock Prices", alpha=0.7)
                plt.plot(ma_50, label="50-Day MA", linestyle="dashed")
                plt.plot(ma_200, label="200-Day MA", linestyle="dashed")
                plt.legend()
                st.pyplot(plt)
                
                st.subheader("📊 Risk vs Return Scatter Plot")
                avg_daily_returns = daily_returns.mean()
                std_deviation = daily_returns.std()
                plt.figure(figsize=(6, 4))
                plt.scatter(std_deviation, avg_daily_returns, c="blue", label="Stocks")
                plt.xlabel("Risk (Std Dev)")
                plt.ylabel("Return (Mean Daily Return)")
                plt.title("Risk vs Return")
                for i, txt in enumerate(tickers_list):
                    plt.annotate(txt, (std_deviation[i], avg_daily_returns[i]))
                st.pyplot(plt)
                
                st.subheader("📄 Stock Data Table")
                st.dataframe(stock_data)
                
                st.subheader("📊 Portfolio Weights Distribution")
                fig, ax = plt.subplots()
                ax.pie(weights_list, labels=tickers_list, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
        except ValueError:
            st.error("⚠ Invalid weight input. Use numeric values separated by commas.")

st.sidebar.markdown("### ℹ About")
st.sidebar.markdown("This app analyzes your stock portfolio by visualizing stock prices, returns, risk-return tradeoff, and performance metrics.")
