# infer_transformer.py

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformer_model import PolypSegformer

INPUT_DIR  = "unlabelled/img"
OUTPUT_DIR = "unlabelled/masks"
MODEL_PATH = "transformer_segformer.pth"
IMG_SIZE   = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = PolypSegformer().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2()
])

images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
print(f"Found {len(images)} images")

for i, img_name in enumerate(images):
    image     = cv2.imread(os.path.join(INPUT_DIR, img_name))
    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    mask = (pred * 255).astype(np.uint8)
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), mask)
    print(f"[{i+1}/{len(images)}] {img_name}")

print(f"Done. Masks saved to {OUTPUT_DIR}/")