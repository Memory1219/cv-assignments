from model import UNet
from dataset import BrianLoader
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
from utils_unet import *
import os


def train_net(net, device, data_path, epochs=100, batch_size=2, lr=0.0001):

    full_dataset = BrianLoader(data_path)
    # Split the data set into training set and verification set
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # define Loss
    criterion = nn.CrossEntropyLoss()
    # initialized to positive infinity
    best_loss = float('inf')

    # Training epochs time
    for epoch in range(epochs):
        # Training mode
        net.train()
        # Initialization
        total_train_loss = 0
        total_train_acc = 0
        ave_train_loss = 0
        ave_train_acc = 0

        # Start training according to batch_size
        for image, label in train_loader:
            label = label.squeeze(1)  # Change the labels shape from [1, 1, 512, 512] to [1, 512, 512]
            optimizer.zero_grad()
            # Copy the data to the device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # Use network parameters to output prediction results
            pred = net(image)
            # calculate loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            total_train_loss += loss.item()
            # calculate accuracy
            acc = calculate_accuracy(pred, label)
            total_train_acc += acc
            # Update parameters
            loss.backward()
            optimizer.step()
        ave_train_loss = total_train_loss / len(train_loader)
        ave_train_acc = total_train_acc / len(train_loader)

        # Verification mode
        net.eval()
        total_test_loss = 0
        total_test_acc = 0
        with torch.no_grad():
            for image, label in test_loader:
                label = label.squeeze(1)  # Change the labels shape from [1, 1, 512, 512] to [1, 512, 512]
                image = image.to(device)
                label = label.to(device)
                output = net(image)
                loss = criterion(output, label)
                total_test_loss += loss.item()
                acc = calculate_accuracy(output, label)
                total_test_acc += acc
        ave_test_loss = total_test_loss / len(test_loader)
        ave_test_acc = total_test_acc / len(test_loader)

        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {ave_train_loss:.4f}, Train Acc: {ave_train_acc:.4f}, Val '
            f'Loss: {ave_test_loss:.4f}, Val Acc: {ave_test_acc:.4f}')

        if ave_test_loss < best_loss:
            save_path = os.path.join('model', 'best_model.pth')
            save_model(net, save_path)
            best_loss = ave_test_loss
            print(f'Model saved to {save_path}')

if __name__ == "__main__":
    # Select the equipment
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Load the network, single channel, classified as 6.
    net = UNet(n_channels=1, n_classes=6)
    # Copy the network to deivce
    net.to(device=device)
    # Specify the address of the training set and start the training.
    data_path = "Brain.mat"
    train_net(net, device, data_path)
