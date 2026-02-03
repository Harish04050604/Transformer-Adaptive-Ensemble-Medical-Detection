import os
import cv2
import torch

class PolypDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        label_path = os.path.join(self.label_dir, img_name + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    W, H = 384, 288
                    x1 = (xc - w/2) * W
                    y1 = (yc - h/2) * H
                    x2 = (xc + w/2) * W
                    y2 = (yc + h/2) * H

                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)   

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target
