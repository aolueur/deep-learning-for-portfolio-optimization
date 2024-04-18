import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=hidden_channels, kernel_size=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=2, padding=1
        )
        self.linear1 = nn.Linear(96, 64)
        self.linear2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward propagation
        """
        c0 = self.conv1(x)
        c1 = F.relu(c0)
        s1 = F.max_pool2d(c1, (2, 2))
        c2 = F.relu(self.conv2(s1))
        s2 = F.max_pool2d(c2, 2)
        s2 = torch.flatten(s2, 1)
        f3 = F.relu(self.linear1(s2))
        f4 = self.linear2(f3)
        output = self.softmax(f4)
        return output


class SharpLoss(nn.Module):
    def __init__(self):
        super(SharpLoss, self).__init__()

    def forward(self, weights, prices):
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
        v = torch.zeros(batch_size)
        logging.debug(f"v: {v}")
        for i in range(batch_size):
            v[i] = torch.dot(weights[i], prices[i])
        logging.debug(f"v: {v}")
        returns = v[1:] / (v[:-1] + 1e-3) - 1
        logging.debug(f"returns: {returns}")
        loss = torch.mean(returns) / torch.std(returns)
        return -loss


if __name__ == '__main__':
    # Test Sharp Loss
    weights = torch.tensor(
        [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    prices = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]).float()
    print(SharpLoss()(weights, prices))
