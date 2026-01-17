import json
import os
import base64
import zlib
import numpy as np
import cv2

ANN_DIR = "labelled/ann"
OUT_DIR = "labelled/bboxes"
os.makedirs(OUT_DIR, exist_ok=True)

def decode_bitmap(bitmap_data, image_height, image_width, origin):
    # Base64 decode
    compressed = base64.b64decode(bitmap_data)

    # Zlib decompress → PNG bytes
    png_bytes = zlib.decompress(compressed)

    # Decode PNG to mask image
    mask = cv2.imdecode(
        np.frombuffer(png_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    # Convert to binary mask
    mask = (mask > 0).astype(np.uint8)

    # Place into full image mask using origin
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    x0, y0 = origin
    h, w = mask.shape
    full_mask[y0:y0+h, x0:x0+w] = mask

    return full_mask



for ann_file in os.listdir(ANN_DIR):
    if not ann_file.endswith(".json"):
        continue

    with open(os.path.join(ANN_DIR, ann_file)) as f:
        data = json.load(f)

    height = data["size"]["height"]
    width = data["size"]["width"]

    boxes = []

    for obj in data["objects"]:
        bitmap = obj["bitmap"]
        mask = decode_bitmap(
            bitmap["data"],
            height,
            width,
            bitmap["origin"]
        )

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        boxes.append([x_min, y_min, x_max, y_max])

    out_file = ann_file.replace(".json", ".txt")
    with open(os.path.join(OUT_DIR, out_file), "w") as f:
        for box in boxes:
            f.write(f"polyp {box[0]} {box[1]} {box[2]} {box[3]}\n")

print("Bounding boxes successfully extracted from Supervisely bitmaps.")
