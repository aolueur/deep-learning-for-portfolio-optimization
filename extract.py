import yfinance as yf
import pandas as pd


def get_data(tickers, start_date, end_date):
    """
    Fetches and prepares financial data for given tickers between start_date and end_date.

    Parameters:
    - tickers (str): List of ticker symbols as strings.
    - start_date (str): The start date for the data retrieval in 'YYYY-MM-DD' format.
    - end_date (str): The end date for the data retrieval in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: A pandas DataFrame with adjusted close prices and calculated returns for the specified tickers.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    tickers_list = (
        tickers.split()
    )  # Assumes tickers are passed as a space-separated string
    adj_close = yf.download(
        ' '.join(tickers_list), start=start_date, end=end_date
    )['Adj Close'].dropna()

    # Create a new DataFrame to store prices and returns
    structured_data = pd.DataFrame()

    for ticker in tickers_list:
        # Add price column
        structured_data[f'{ticker}_Price'] = adj_close[ticker]

        # Calculate and add return column immediately after price column
        structured_data[f'{ticker}_Return'] = adj_close[ticker].pct_change()

    # Drop the first row since it will have NaN values for returns
    structured_data.dropna(inplace=True)

    return structured_data
