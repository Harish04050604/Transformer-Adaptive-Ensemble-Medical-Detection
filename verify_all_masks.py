#verify_all_masks.py

import os
import cv2
import numpy as np

INPUT_DIR  = os.path.join("unlabelled", "img")
MASK_DIR   = os.path.join("unlabelled", "masks")
OUTPUT_DIR = os.path.join("unlabelled", "verification")

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])
print(f"Generating verification images for {len(images)} images...")

for i, img_name in enumerate(images):
    image = cv2.imread(os.path.join(INPUT_DIR, img_name))
    mask  = cv2.imread(os.path.join(MASK_DIR, img_name), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {img_name} - file missing")
        continue

    overlay = image.copy()
    overlay[mask > 127] = (0, 255, 0)
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    mask_bgr    = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comparison  = np.hstack([image, mask_bgr, blended])

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), comparison)
    print(f"[{i+1}/{len(images)}] {img_name}")

print(f"Done. Saved to {OUTPUT_DIR}/")