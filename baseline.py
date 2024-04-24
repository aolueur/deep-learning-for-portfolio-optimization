import numpy as np
import pandas as pd


def calculate_portfolio_returns(data_train, tickers):
    """
    Calculates the equal-weighted portfolio returns from individual stock returns.

    Parameters:
    - data_train (DataFrame): DataFrame containing the stock returns. It can be obtained after running get_data function
    - tickers (str): List of ticker symbols as strings.

    Returns:
    - Series: A pandas Series containing the portfolio returns.
    """
    tickers_list = (
        tickers.split()
    )  # Ensures we have a list of tickers from a string
    # Calculate the weights as an array where each ticker has equal weight
    weights = np.array([1 / len(tickers_list)] * len(tickers_list))
    # Filter the DataFrame for return columns and calculate the portfolio returns using dot product
    portfolio_ret = data_train.filter(like='_Return').dot(weights)
    return portfolio_ret


def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio of the portfolio returns.

    Parameters:
    - portfolio_returns (Series): A pandas Series of portfolio returns.
      It can be obtained from calculate_portfolio_returns function
    - risk_free_rate (float): The risk-free rate of return. Assume risk-free rate is 0 here for simplicity

    Returns:
    - float: The Sharpe ratio of the portfolio.
    """
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    baseline_SR = (mean_return - risk_free_rate) / std_dev
    return baseline_SR
