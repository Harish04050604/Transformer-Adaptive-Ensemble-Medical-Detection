import torch
import cv2
import os
from torchvision import transforms
from lrpm_ham_model import LRPM_HAM_Detector

# ---- CONFIG ----
IMAGE_PATH = "unlabelled/img/1.png"   # change as needed
MODEL_PATH = "lrpm_ham_detector.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load Model ----
model = LRPM_HAM_Detector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---- Load Image ----
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.ToTensor()
img_tensor = transform(img).to(device)

input_shape = img_tensor.shape  # C,H,W

# ---- Extract Feature Map ----
with torch.no_grad():
    x = img_tensor.unsqueeze(0)  # add batch dim
    feature_map = model.backbone(x)
    feature_map = model.ham(feature_map)

feature_shape = feature_map.shape  # B,C,H,W

# ---- Print Details ----
print("\n===== INPUT DETAILS =====")
print(f"Input Tensor Shape (C,H,W): {input_shape}")
print(f"Batch Shape: {x.shape}")

print("\n===== FEATURE MAP DETAILS =====")
print(f"Feature Map Shape (B,C,H,W): {feature_shape}")
print(f"Channels: {feature_shape[1]}")
print(f"Spatial Size: {feature_shape[2]} x {feature_shape[3]}")

# Spatial reduction
reduction_h = input_shape[1] // feature_shape[2]
reduction_w = input_shape[2] // feature_shape[3]

print("\n===== SPATIAL REDUCTION =====")
print(f"Height Reduction Factor: {reduction_h}x")
print(f"Width Reduction Factor: {reduction_w}x")

