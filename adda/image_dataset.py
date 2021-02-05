#!/usr/bin/env python
# coding: utf-8

# In[52]:


from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob
import pandas as pd
import torch
import random


# In[124]:


class ImageDataset(Dataset):
    def __init__(self, mode, dataset_name, is_target=False):
        assert mode in ["train", "val", "test"]
        assert dataset_name in ["mnistm", "svhn", "usps"]

        self.mode = mode
        self.dataset_name = dataset_name
#         self.is_target = is_target

        self.init_img_paths()
        self.init_labels()

        self.len = len(self.img_paths)

    def init_img_paths(self):
        mode = "train" if self.mode == "val" else self.mode

        img_path = "../hw3_data/digits/{}/{}/".format(self.dataset_name, mode)
        img_paths = sorted(glob.glob(img_path + '*.png'))

        if mode == "train":
            train_len = int(0.9 * len(img_paths))
            img_paths = img_paths[:train_len] if self.mode == "train" else img_paths[train_len:]

        self.img_paths = img_paths

    def init_labels(self):
        mode = "train" if self.mode == "val" else self.mode

        label_csv_path = "../hw3_data/digits/{}/{}.csv".format(
            self.dataset_name, mode)
        df = pd.read_csv(label_csv_path)
        labels = df['label'].to_numpy()

        if mode == "train":
            train_len = int(0.9 * len(labels))
            labels = labels[:train_len] if self.mode == "train" else labels[train_len:]

        self.labels = labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split('/')[-1]

        # get image tensor
        image = Image.open(img_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # data augmentation
#         if self.mode == "train" and not self.is_target:
#             transform = T.Compose([
#                     T.RandomApply([
#                         T.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),
#                         T.RandomRotation(degrees=45),
#                         T.RandomCrop(26),
#                         T.Resize(28),
#                     ], p=0.5),
#                     T.ToTensor(),
#                     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ])

        image_tensor = transform(image)

        # get label tensor
        label = self.labels[idx]
        label_tensor = torch.tensor(int(label))

        return image_tensor, label_tensor

    def __len__(self):
        return self.len
