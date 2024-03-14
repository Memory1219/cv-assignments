import torch
import os
from data_process import BrainDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model import UNet


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model


full_dataset = BrainDataset('Brain.mat')
# 将数据集分割为训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 定义数据加载时的一些参数
batch_size = 32

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=1, n_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
best_val_acc = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_acc = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, labels)
        total_train_acc += acc
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_acc / len(train_loader)

    # 验证循环
    model.eval()
    total_val_loss = 0
    total_val_acc = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            acc = calculate_accuracy(outputs, labels)
            total_val_acc += acc

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val '
          f'Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')

    # 如果这个周期的验证损失是迄今为止最好的，保存模型
    if avg_val_loss < best_val_loss:
        save_path = os.path.join('/model', 'best_model.pth')
        save_model(model, save_path)
        best_val_loss = avg_val_loss
        print(f'Model saved to {save_path}')
