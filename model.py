import numpy as np
import torch
import torch.nn as nn
import logging

logging.basicConfig(format="%(asctime)s-%(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")


class ConvNet(nn.Module):
    """
    Convolutional Neural Network

    k: loopback window size
    num_asset: number of assets
    num_fields: number of fields

    We treat the loopback window as channels.
    The num_asset and num_fields are treated as the height and width of the "image".
    Input data is of the shape (batch_size, k, num_asset, num_fields)
    """

    def __init__(self, input_channels=50, hidden_channels=16, output_dim=4):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(
                out_channels=hidden_channels, kernel_size=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(
                out_channels=hidden_channels, kernel_size=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LazyConv2d(
                out_channels=hidden_channels, kernel_size=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward propagation
        """
        return self.net(x)


def sharp_loss(weights, prices):
    """
    Compute the sharp ratio loss
    Args:
        weights: (batch_size, num_assets). weights of the portfolio.
        prices: (batch_size, num_assets). historical prices of all assets
    """
    logging.debug(f"weights shape: {weights.shape}")
    logging.debug(f"prices shape: {prices.shape}")
    batch_size = weights.shape[0]
    assert weights.shape == prices.shape, "Shapes do not match"
    v = []
    for i in range(batch_size):
        v.append(torch.dot(weights[i], prices[i]))
    v = torch.tensor(v).float()
    returns = v[1:] / v[:-1] - 1
    loss = torch.mean(returns) / torch.std(returns)
    return -loss
