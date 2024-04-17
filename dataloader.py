import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from extract import get_data
from hyperparams import tickers, train_start_date, train_end_date, validation_start_date, validation_end_date
import logging

logging.basicConfig(format="%(asctime)s-%(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
# logging.getLogger().setLevel(logging.DEBUG)


# placeholder
class IdentityTransform(nn.Module):
    def __init__(self):
        super(IdentityTransform, self).__init__()

    def forward(self, x):
        return x


class PortfolioDataset(Dataset):
    def __init__(
        self,
        data,
        lookback_window=50,
        num_assets=4,
        num_features=2,
        transform=None,
    ):
        """
        Initializes the dataset with financial time-series data.

        :param data: Pandas DataFrame containing the financial data. Columns should be ordered
                     as features for each asset, repeating for each asset.
        :param lookback_window: 50. The number of historical days to include in each sample.
        :param num_assets: Number of assets. set to 4.
        :param num_features: Number of features per asset. set to 2. prices and return
        :param transform: Optional transform to be applied to each sample.
        """
        self.data = data
        self.lookback_window = lookback_window
        self.num_assets = num_assets
        self.num_features = num_features
        self.transform = transform

    def __len__(self):

        # Ensuring we have enough data for at least one lookback window
        return len(self.data) - self.lookback_window + 1

    def __getitem__(self, idx):

        start_idx = idx
        end_idx = idx + self.lookback_window
        logging.debug(f"start_idx: {start_idx}, end_idx: {end_idx}")
        x = self.data.iloc[start_idx:end_idx, :].values
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Reshape x to have shape (lookback_window, num_assets, num_features)
        x_tensor = x_tensor.view(
            self.lookback_window, self.num_assets, self.num_features
        )
        logging.debug("x_tensor shape: {}".format(x_tensor.shape))
        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        # add label y to document start-end dat of each xt
        y = x_tensor[-1, :, 0]

        return x_tensor, y


data_train = get_data(tickers, train_start_date, train_end_date)
data_val = get_data(tickers, validation_start_date, validation_end_date)

identity_transform = IdentityTransform()
train_dataset = PortfolioDataset(
    data_train,
    lookback_window=50,
    num_assets=4,
    num_features=2,
    transform=identity_transform,
)
val_dataset = PortfolioDataset(
    data_val,
    lookback_window=50,
    num_assets=4,
    num_features=2,
    transform=identity_transform,
)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, drop_last=True)


if __name__ == '__main__':
    print(train_dataset[0][1])
