import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class ConvNet(nn.Module):
    """
    Convolutional Neural Network

    k: loopback window size
    num_asset: number of assets
    num_fields: number of fields

    We treat the loopback window as channels. The num_asset and num_fields are treated as the height and width of the "image". Input data is of the shape (batch_size, k, num_asset, num_fields)
    """

    def __init__(self, input_channels=50, hidden_channels=16, output_dim=4):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=hidden_channels,
                          kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(out_channels=hidden_channels,
                          kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(out_channels=hidden_channels,
                          kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward propagation
        """
        return self.net(x)


def sharp_loss(weights, prices, idx):
    """
    Compute the sharp ratio loss
    Args:
        weights: (batch_size, num_assets). weights of the portfolio
        prices: (num_assets, time_horizion). historical prices of all assets
        idx: an array of size batch_size. Every element is a tuple (start_idx, end_idx). For example, for weights w_t, the corresponding time period is (t - 50, t - 1).
    """
    batch_size = weights.shape[0]
    loss = 0
    for i in range(batch_size):
        start_idx, end_idx = idx[i]
        returns = prices[:, start_idx:end_idx].T
        portfolio_values = np.sum(returns * weights[i], axis=1)
        portfolio_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        sharp_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns)
        loss += -sharp_ratio
    return loss / batch_size
