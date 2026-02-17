import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from lrpm_ham_model import LRPM_HAM_Detector

# -----------------------------
# Paths
# -----------------------------
IMG_DIR = "labelled/img"
OUT_DIR = "labelled/feature_visuals"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load model
# -----------------------------
model = LRPM_HAM_Detector().to(device)
model.eval()

transform = transforms.ToTensor()

with torch.no_grad():
    for img_name in os.listdir(IMG_DIR):

        if not img_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(IMG_DIR, img_name)

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        img_tensor = transform(img).to(device)

        # Extract feature map
        feature_map = model.extract_feature_maps([img_tensor])
        fm = feature_map[0]

        # Average channels → 2D map
        fm_combined = torch.mean(fm, dim=0).cpu().numpy()

        # Normalize
        fm_combined -= fm_combined.min()
        fm_combined /= (fm_combined.max() + 1e-8)

        # Resize to original size
        fm_resized = cv2.resize(fm_combined, (W, H))

        # -----------------------------
        # Create side-by-side figure
        # -----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(fm_resized, cmap="jet")
        axes[1].set_title("Feature Map")
        axes[1].axis("off")

        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, img_name)
        plt.savefig(out_path)
        plt.close()

        print(f"✅ Saved visualization: {img_name}")

print("\n🎉 All feature visualizations generated successfully.")