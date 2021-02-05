#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import pandas as pd
import torch


class ReproduceDataset(Dataset):
    def __init__(self, test_path):
        self.test_path = test_path

        self.init_img_paths()
        self.len = len(self.img_paths)

    def init_img_paths(self):
        img_paths = sorted(glob.glob(self.test_path + '/*.png'))
        self.img_paths = img_paths

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split('/')[-1]

        # get image tensor
        image = Image.open(img_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        image_tensor = transform(image)
        return image_tensor

    def __len__(self):
        return self.len
