import os

import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch


class MetalPlateDataset(Dataset):
    def __init__(self, root_dir, size=(256, 256), exts=("png",), dataset_mean=None):
        self.root_dir = root_dir
        self.size = size
        self.exts = exts
        # collect all image paths
        if 'train' in self.root_dir:
            self.image_paths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith(tuple(exts))
            ]
            self.image_paths.sort()  # optional: keep deterministic order

        else:
            excluded = {'.', '..', '__MACOSX', '.DS_Store', '__pycache__', '.git'}
            test_paths = [dir for dir in os.listdir(self.root_dir) if dir not in excluded]
            self.image_paths = []
            for test_path in test_paths:
                current_test_paths = [
                    os.path.join(root_dir, test_path, f)
                    for f in os.listdir(os.path.join(root_dir, test_path))
                    if f.lower().endswith(tuple(exts))
                ]
                self.image_paths.extend(current_test_paths)

            self.image_paths.sort()  # optional: keep deterministic order

        
        ### compute the dataset mean for preprocessing
        if dataset_mean == None:
            ## for train
            pytorch_images = []

            for path in self.image_paths:
                img = cv2.imread(path, cv2.IMREAD_COLOR)  # shape [H,W,3], BGR
                img = cv2.resize(img, self.size)
                img_tensor = TF.to_tensor(img)
                pytorch_images.append(img_tensor)

            all_images_tensor = torch.stack(pytorch_images)
            self.dataset_mean = torch.mean(all_images_tensor)
            del pytorch_images

        else:
            ## for test
            self.dataset_mean = dataset_mean

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        label = 0
        if 'good' in path:
            label = 1

        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        img = cv2.resize(img, self.size)
        img_tensor = TF.to_tensor(img)
        
        #### mean subtraction
        img_tensor -= self.dataset_mean
        return img_tensor, label, self.dataset_mean   # return numpy array + optional path