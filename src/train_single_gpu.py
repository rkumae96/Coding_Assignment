import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from your_model_file import GPT2, GPT2Config  # Import your model

def train_single_gpu(model, dataloader, loss_fn, optimizer, device):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2(GPT2Config())
dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_single_gpu(model, dataloader, loss_fn, optimizer, device)
