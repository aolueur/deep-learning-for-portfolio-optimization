import torch
from model import sharp_loss
from dataloader import train_loader, val_loader
from model import ConvNet
import logging
import tqdm
from hyperparams import num_epochs, lr, print_freq, input_channels, hidden_channels

logging.basicConfig(format="%(asctime)s-%(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
# logging.getLogger().setLevel(logging.DEBUG)


def train(
    train_loader, val_loader, model, num_epochs, lr=1e-1, print_freq=100
):
    """
    Model training loop
    """

    logging.info("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        logging.info(f"Trainging epoch {epoch}")
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            weights = model(x)
            loss = sharp_loss(weights, y)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % print_freq == 0:
                print(f'epcho {epoch} loss {loss.item()}')

        logging.info(f"Validating epoch {epoch}")
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                weights = model(x)
                loss = sharp_loss(weights, y)
            if (batch_idx + 1) % print_freq == 0:
                print(f'epcho {epoch} test loss {loss.item()}')
    return model


model = ConvNet(input_channels, hidden_channels, output_dim=4)

train(train_loader, val_loader, model, num_epochs, lr, print_freq)
