import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioDataset(Dataset):
    def __init__(self, data, lookback_window=50):
        self.data = data
        self.lookback_window = lookback_window
        
    def __len__(self):
        #length of dataset
        return len(self.data) - self.lookback_window
    
    def __getitem__(self, idx):
        #retrieves a single sample from the dataset
        start_idx = idx
        end_idx = idx + self.lookback_window
        
        input_data = self.data.iloc[start_idx:end_idx]
        input_data = self.preprocess_data(input_data)
        
        return torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    def preprocess_data(self, data):
        preprocessed_data = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)
        return preprocessed_data

def load_data(csv_file):
    data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return data

def create_datasets_and_loaders(data, train_start_date, train_end_date, test_start_date, test_end_date, lookback_window=50, batch_size=64):
    train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
    test_data = data[(data.index >= test_start_date) & (data.index <= test_end_date)]
    
    train_dataset = PortfolioDataset(train_data, lookback_window)
    test_dataset = PortfolioDataset(test_data, lookback_window)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader


#get data for source/file

csv_file = 'CSV_file'
data = load_data(csv_file)

train_start_date = '2009-01-01'
train_end_date = '2013-12-31'
test_start_date = '2014-01-01'
test_end_date = '2024-03-15'

train_dataset, test_dataset, train_loader, test_loader = create_datasets_and_loaders(
    data, train_start_date, train_end_date, test_start_date, test_end_date, lookback_window=50, batch_size=32
)

# Create an instance of the ConvNet model
model = ConvNet(input_channels=50, hidden_channels=16, output_dim=4)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = sharp_loss

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss = criterion(outputs, data.values, train_dataset.lookback_window)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print the loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Generate portfolio allocations for the test data
portfolio_allocations = []
with torch.no_grad():
    for batch in test_loader:
        allocations = model(batch)
        portfolio_allocations.extend(allocations.numpy())

# Calculate portfolio returns
portfolio_returns = []
for i in range(len(test_dataset)):
    allocations = portfolio_allocations[i]
    asset_returns = test_dataset.data.iloc[i + test_dataset.lookback_window].filter(like='Return').values
    portfolio_return = np.sum(allocations * asset_returns)
    portfolio_returns.append(portfolio_return)

# Aggregate portfolio returns by year
yearly_returns = {}
for date, returns in zip(test_dataset.data.index[test_dataset.lookback_window:], portfolio_returns):
    year = date.year
    if year not in yearly_returns:
        yearly_returns[year] = []
    yearly_returns[year].append(returns)

avg_yearly_returns = {year: np.mean(returns) for year, returns in yearly_returns.items()}

# Plot the yearly portfolio returns
years = list(avg_yearly_returns.keys())
returns = list(avg_yearly_returns.values())

plt.figure(figsize=(10, 6))
plt.bar(years, returns)
plt.xlabel('Year')
plt.ylabel('Average Portfolio Return')
plt.title('Yearly Portfolio Returns')
plt.xticks(years)
plt.show()