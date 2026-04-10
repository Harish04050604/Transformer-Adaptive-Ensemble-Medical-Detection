import os
import json
import cv2
import torch
import random
import base64
import zlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ram_model import RAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RAMDataset(Dataset):
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def decode_mask(self, bitmap):
        data = base64.b64decode(bitmap["data"])
        data = zlib.decompress(data)
        mask = np.frombuffer(data, dtype=np.uint8)

        size = int(np.sqrt(len(mask)))
        mask = mask[:size*size].reshape(size, size)

        return mask, bitmap["origin"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + ".json")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape

        # 🔥 build full mask
        full_mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)

            for obj in data["objects"]:
                mask, (ox, oy) = self.decode_mask(obj["bitmap"])
                mh, mw = mask.shape

                full_mask[oy:oy+mh, ox:ox+mw] = np.maximum(
                    full_mask[oy:oy+mh, ox:ox+mw], mask
                )

        mask_area = np.sum(full_mask > 0)

        # 🔥 SMART SAMPLING
        if random.random() < 0.7 and mask_area > 0:
            ys, xs = np.where(full_mask > 0)

            idx_rand = random.randint(0, len(xs)-1)
            cx, cy = xs[idx_rand], ys[idx_rand]

            bw = random.randint(64, 128)
            bh = random.randint(64, 128)

            x1 = max(0, cx - bw//2)
            y1 = max(0, cy - bh//2)

            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)
        else:
            bw = random.randint(64, 160)
            bh = random.randint(64, 160)

            x1 = random.randint(0, w - bw)
            y1 = random.randint(0, h - bh)

            x2 = x1 + bw
            y2 = y1 + bh

        crop = image[y1:y2, x1:x2]
        mask_crop = full_mask[y1:y2, x1:x2]

        crop = cv2.resize(crop, (128,128))
        crop = torch.tensor(crop/255.0, dtype=torch.float32).permute(2,0,1)

        # 🔥 FINAL CORRECT LABEL
        intersection = np.sum(mask_crop > 0)

        if mask_area == 0:
            score = 0.0
        else:
            score = intersection / (mask_area + 1e-6)

        score = min(score, 1.0)

        # 🔥 spread values (IMPORTANT)
        score = score ** 0.5

        return crop, torch.tensor([score], dtype=torch.float32)


dataset = RAMDataset("labelled/img", "labelled/ann")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = RAM().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "ram_model.pth")
print("RAM trained properly")