from hyperparams import *
from model import ConvNet
import torch
from baseline import calculate_portfolio_returns, calculate_sharpe_ratio
from extract import get_data
from dataloader import PortfolioDataset, IdentityTransform
from torch.utils.data import DataLoader
from model import SharpLoss

# Load the ConvNet model
model = ConvNet(input_channels, hidden_channels, output_dim=4)
model.load_state_dict(torch.load("model.pth"))

identity_transform = IdentityTransform()

start_date = '2023-01-01'
end_date = '2023-12-31'

# Load the data
test_data = get_data(tickers, start_date, end_date)

# Evaluate the CNN model on the test set
test_dataset = PortfolioDataset(
    test_data,
    lookback_window=50,
    num_assets=4,
    num_features=2,
    transform=identity_transform,
)

test_loader = DataLoader(
    test_dataset, batch_size=len(test_dataset), shuffle=False)
sharp_loss = SharpLoss()

model.eval()

X, y = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X)
    conv_loss = sharp_loss(y_pred, y)

conv_sharpe = -conv_loss.item()

# Evaluate the baseline model on the test set
returns = calculate_portfolio_returns(test_data, tickers)
baseline_sharpe = calculate_sharpe_ratio(returns)

print("CNN Sharpe Ratio:", conv_sharpe)
print("Baseline Sharpe Ratio:", baseline_sharpe)
