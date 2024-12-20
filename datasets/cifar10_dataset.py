import torch
from torchvision import datasets
import numpy as np

class CIFAR10Dataset:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.data[idx][1]
        
        # Convert PIL Image to numpy array
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label 