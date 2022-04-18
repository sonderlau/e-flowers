from torch import utils,from_numpy
from PIL import Image
import numpy as np
import os

"""
自定义的鲜花数据集模块
"""


class FlowerDataset(utils.data.Dataset):
    def __init__(self, dataset_path='./dataset/processed', transform=None, train=True):
        self.transforms = transform

        if train:
            self.dataset_path = os.path.join(dataset_path, "train/")
        else:
            self.dataset_path = os.path.join(dataset_path, "validation/")
        
        self.images = [] 
        
        for image in os.listdir(self.dataset_path):
            self.images.append(os.path.join(self.dataset_path, image))

        self.size = len(os.listdir(self.dataset_path))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = img_path.split("_")[1]
        return self.transforms(img), from_numpy(np.array([int(label) - 1]))

# training_data = FlowerDataset('./dataset/processed', transform=ToTensor())


# print(training_data.images)