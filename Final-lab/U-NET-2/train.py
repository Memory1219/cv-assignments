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
    # 将数据集分割为训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 初始化
        total_train_loss = 0
        total_train_acc = 0
        ave_train_loss = 0
        ave_train_acc = 0

        # 按照batch_size开始训练
        for image, label in train_loader:
            label = label.squeeze(1)  # 将labels形状从[1, 1, 512, 512]改为[1, 512, 512]
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            total_train_loss += loss.item()
            # 计算accuracy
            acc = calculate_accuracy(pred, label)
            total_train_acc += acc
            # 保存loss值最小的网络参数
            # if loss < best_loss:
            #     best_loss = loss
            #     torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        ave_train_loss = total_train_loss / len(train_loader)
        ave_train_acc = total_train_acc / len(train_loader)
        # print(f'Epoch {epoch + 1}, Loss: {ave_train_loss}, Accuracy: {ave_train_acc}%')

        # 保存最佳模型
        # if ave_train_loss < best_loss:
        #     best_loss = ave_train_loss
        #     torch.save(net.state_dict(), 'best_model.pth')

        net.eval()
        total_test_loss = 0
        total_test_acc = 0
        with torch.no_grad():
            for image, label in test_loader:
                label = label.squeeze(1)  # 将labels形状从[1, 1, 512, 512]改为[1, 512, 512]
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
            best_val_loss = ave_test_loss
            print(f'Model saved to {save_path}')

if __name__ == "__main__":
    # 选择设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=6)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "Brain.mat"
    train_net(net, device, data_path)
