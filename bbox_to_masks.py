#bbox_to_masks.py

import os
import cv2
import json
import numpy as np
import base64
import zlib
from PIL import Image
import io

IMAGE_DIR = os.path.join("labelled", "img")
ANN_DIR = "ann"
MASK_DIR = "labelled_masks"

os.makedirs(MASK_DIR, exist_ok=True)

available_images = set(os.listdir(IMAGE_DIR))

for ann_file in os.listdir(ANN_DIR):
    if not ann_file.endswith(".json"):
        continue

    img_name = os.path.splitext(ann_file)[0]  # "10.png.json" -> "10.png"

    if img_name not in available_images:
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read image: {img_path}")
        continue

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    ann_path = os.path.join(ANN_DIR, ann_file)
    with open(ann_path, "r") as f:
        ann = json.load(f)

    for obj in ann.get("objects", []):
        if obj.get("classTitle") != "polyp":
            continue

        if obj.get("geometryType") == "bitmap":
            bitmap_data = obj["bitmap"]["data"]
            origin = obj["bitmap"]["origin"]
            ox, oy = origin[0], origin[1]

            # Fix base64 padding
            padding = 4 - len(bitmap_data) % 4
            if padding != 4:
                bitmap_data += "=" * padding

            try:
                # Step 1: base64 decode
                raw = base64.b64decode(bitmap_data)
                # Step 2: zlib decompress
                decompressed = zlib.decompress(raw)
                # Step 3: open as PNG
                bitmap_img = Image.open(io.BytesIO(decompressed)).convert("L")
                bitmap_arr = (np.array(bitmap_img) > 0).astype(np.uint8) * 255
            except Exception as e:
                print(f"Decode failed for {ann_file}: {e}")
                continue

            bh, bw = bitmap_arr.shape
            x1, y1 = ox, oy
            x2, y2 = ox + bw, oy + bh

            # Clamp to image bounds
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)

            mask[y1c:y2c, x1c:x2c] = bitmap_arr[y1c - y1:y2c - y1, x1c - x1:x2c - x1]

        elif obj.get("geometryType") == "rectangle":
            pts = obj["points"]["exterior"]
            x1, y1 = int(pts[0][0]), int(pts[0][1])
            x2, y2 = int(pts[1][0]), int(pts[1][1])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 255

    mask_save_path = os.path.join(MASK_DIR, img_name)
    cv2.imwrite(mask_save_path, mask)
    print(f"Saved mask: {img_name}")

print(f"Done. Total masks saved to {MASK_DIR}/")