#segmentation_dataset.py

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMG_SIZE = 256

class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file, augment=False):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir

        with open(split_file, "r") as f:
            self.images = [l.strip() for l in f.readlines()]

        if augment:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask  = cv2.imread(os.path.join(self.mask_dir, img_name),
                           cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].unsqueeze(0)