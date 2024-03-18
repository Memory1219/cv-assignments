# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import numpy as np

# data = loadmat('Brain.mat')
# T1_images = data['T1']
# labels = data['label']
# T1_tensor = torch.tensor(T1_images, dtype=torch.float32).unsqueeze(1)  # Add a channel dimension
# labels_tensor = torch.tensor(labels, dtype=torch.long)

# Define Dataset class
class BrainDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# # Create training and test datasets
# train_dataset = BrainDataset(T1_tensor[:8], labels_tensor[:8])
# test_dataset = BrainDataset(T1_tensor[8:], labels_tensor[8:])
#
# # Create Data Loader
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# # Check Data Loader Output
# next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape