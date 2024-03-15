from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import cv2
import random


def data_process(image, label):
    image = torch.tensor(image, dtype=torch.float32)
    # Label: No need to add a channel dimension for labels in segmentation tasks
    label = torch.tensor(label, dtype=torch.long)
    # Calculate padding
    # For height
    top_padding = (512 - image.shape[0]) // 2
    bottom_padding = 512 - image.shape[0] - top_padding
    # For width
    left_padding = (512 - image.shape[1]) // 2
    right_padding = 512 - image.shape[1] - left_padding

    # Pad the image and label
    image = F.pad(image.unsqueeze(0), (left_padding, right_padding, top_padding, bottom_padding))
    label = F.pad(label.unsqueeze(0), (left_padding, right_padding, top_padding, bottom_padding),
                  value=-1)  # Assuming -1 is an ignore index for labels

    image = np.asarray(image) #(1, 512, 512)
    label = np.asarray(label) #(1, 512, 512)
    return image, label

class BrianLoader(Dataset):
    def __init__(self, mat_file_path):
        # Load data
        self.data = loadmat(mat_file_path)
        self.images = self.data['T1']
        self.labels = self.data['label']

    def __len__(self):
        return self.images.shape[2]

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, idx):
        image = self.images[:, :, idx]
        label = self.labels[:, :, idx]

        image, label = data_process(image, label)

        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label



if __name__ == "__main__":
    brian_data = BrianLoader("Brain.mat")
    print("数据个数：", len(brian_data))
    train_loader = torch.utils.data.DataLoader(dataset=brian_data,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)

    brian_data = loadmat('Brain.mat')
    images = brian_data['T1']
    labels = brian_data['label']
    image1, label1 = data_process(images[:, :, 0], labels[:, :, 0])
    print(image1.shape)
    print(label1.shape)

