import torch
import keras
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rgb_to_grayscale

def one_hot(y: np.ndarray, num_classes: int):
    return np.eye(num_classes)[y]

class Cifar10Dataset(Dataset):
    def __init__(self, data, labels, classes, image_type, maxi, mini):
        super().__init__()

        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
        if image_type:
            data = rgb_to_grayscale(data)
        self.data = data
        self.labels = one_hot(labels, len(classes)).astype(np.float32)
        self.classes = classes
        self.max = maxi
        self.min = mini

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        sample = (sample - self.min)/(self.max - self.min)
        return sample, label