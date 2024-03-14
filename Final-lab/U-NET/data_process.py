from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class BrainDataset(Dataset):
    def __init__(self, mat_file_path):
        # Load data
        self.data = loadmat(mat_file_path)
        self.images = self.data['T1']
        self.labels = self.data['label']

    def __len__(self):
        return self.images.shape[2]

    def __getitem__(self, idx):
        image = self.images[:, :, idx]
        label = self.labels[:, :, idx]
        # Convert to PyTorch tensors
        # Image: Add a channel dimension to make it (1, 362, 434)
        # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
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

        # Now image and label are (1, 512, 512), label is padded with -1 to be ignored in loss calculation if necessary
        return image, label



