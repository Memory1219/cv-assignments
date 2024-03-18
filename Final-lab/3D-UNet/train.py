import torch
import os

from scipy.io import loadmat

from data_process import BrainDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import UNet3D
from torch import optim


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


# loading
data = loadmat('Brain.mat')
T1_images = data['T1']
labels = data['label']

# data preparation steps
T1_tensor = torch.tensor(T1_images, dtype=torch.float32).unsqueeze(0)  # 增加一个通道维度
labels_tensor = torch.tensor(labels, dtype=torch.long).unsqueeze(0)

# Create training and test datasets
train_dataset = BrainDataset(T1_tensor[:8], labels_tensor[:8])
test_dataset = BrainDataset(T1_tensor[8:], labels_tensor[8:])

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Set up device
device = torch.device('cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet3D(n_channels=1, n_classes=6)
# loading model to continue training if there is already a model
model.load_state_dict(torch.load('model/best_model.pth', map_location=device))
model = model.to(device)

# Defining Loss Functions and Optimizers
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
num_epochs = 20
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.unsqueeze(0)
        images = images.to(device)
        labels = labels.to(device)

        # forward propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    if running_loss < best_loss:
        save_path = os.path.join('model', 'best_model.pth')
        save_model(model, save_path)
        best_loss = running_loss
        print(f'Model saved to {save_path}')
print("Finished Training")
