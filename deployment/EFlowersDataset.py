from torch import utils
from PIL import Image
import os


class FlowerDataset(utils.data.Dataset):
    def __init__(self, dataset_path, transform=None, train=True):

        self.transforms = transform

        if train:
            self.dataset_path = os.path.join(dataset_path, "train/")
        else:
            self.dataset_path = os.path.join(dataset_path, "validation/")
        print(self.dataset_path)
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
        return self.transforms(img), label
