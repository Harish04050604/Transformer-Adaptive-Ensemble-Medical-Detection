import os
import cv2
import numpy as np

# --------------------
# Paths
# --------------------
CLASS_PRED_DIR = "unlabelled/class_prediction"
MASK_DIR = "unlabelled/masks"
OUTPUT_DIR = "unlabelled/pseudo_labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image size (change if needed)
IMG_WIDTH = 384
IMG_HEIGHT = 288

# Confidence threshold
CONF_THRESHOLD = 0.5

# --------------------
# Process each file
# --------------------
for file_name in os.listdir(CLASS_PRED_DIR):

    if not file_name.endswith(".txt"):
        continue

    base_name = file_name.replace(".txt", "")
    class_path = os.path.join(CLASS_PRED_DIR, file_name)
    mask_path = os.path.join(MASK_DIR, base_name + ".png")

    print(f"\nProcessing: {base_name}")

    # --------------------
    # Read prediction file
    # --------------------
    with open(class_path, "r") as f:
        content = f.read().strip()

    # Case 1: No detection
    if content == "N/A":
        print("Skipped (No detection)")
        continue

    # --------------------
    # Extract score (robust)
    # --------------------
    parts = content.split()

    try:
        if len(parts) == 1:
            score = float(parts[0])
        else:
            score = float(parts[-1])  # take last value
    except:
        print("Invalid format, skipping")
        continue

    print(f"Confidence score: {score}")

    # --------------------
    # Filter low confidence
    # --------------------
    if score < CONF_THRESHOLD:
        print("Skipped (Low confidence)")
        continue

    # --------------------
    # Check mask exists
    # --------------------
    if not os.path.exists(mask_path):
        print("Mask not found, skipping")
        continue

    # --------------------
    # Load mask
    # --------------------
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask loading failed, skipping")
        continue

    # Binary threshold
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # --------------------
    # Find contours
    # --------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("No object found in mask")
        continue

    # Take largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    # --------------------
    # Convert to YOLO format
    # --------------------
    x_center = (x + w / 2) / IMG_WIDTH
    y_center = (y + h / 2) / IMG_HEIGHT
    width = w / IMG_WIDTH
    height = h / IMG_HEIGHT

    class_id = 0  # polyp

    # --------------------
    # Save pseudo label
    # --------------------
    output_path = os.path.join(OUTPUT_DIR, base_name + ".txt")

    with open(output_path, "w") as f:
        f.write(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        )

    print(f"Pseudo label saved → {output_path}")

# --------------------
print("\nPseudo-label generation completed successfully!")