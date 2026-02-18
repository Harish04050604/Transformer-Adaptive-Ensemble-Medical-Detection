import os
import json
import base64
import zlib
import cv2
import numpy as np

# ================= PATHS =================
IMG_DIR = "img"
ANN_DIR = "ann"
OUT_DIR = "gt_bbox_visuals"

os.makedirs(OUT_DIR, exist_ok=True)

# ================= BITMAP DECODER =================
def decode_bitmap(bitmap_data):
    compressed = base64.b64decode(bitmap_data)
    png_bytes = zlib.decompress(compressed)

    mask = cv2.imdecode(
        np.frombuffer(png_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    return (mask > 0).astype(np.uint8)

# ================= MAIN LOOP =================
for ann_file in os.listdir(ANN_DIR):
    if not ann_file.endswith(".json"):
        continue

    image_name = ann_file.replace(".json", "")
    image_path = os.path.join(IMG_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"❌ Image missing: {image_name}")
        continue

    # Load image
    image = cv2.imread(image_path)

    # Load annotation
    with open(os.path.join(ANN_DIR, ann_file)) as f:
        data = json.load(f)

    for obj in data["objects"]:
        bitmap = obj["bitmap"]
        mask = decode_bitmap(bitmap["data"])

        origin_x, origin_y = bitmap["origin"]
        h, w = mask.shape

        # Get bbox from mask
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        x_min = origin_x + xs.min()
        x_max = origin_x + xs.max()
        y_min = origin_y + ys.min()
        y_max = origin_y + ys.max()

        # Draw GT bounding box (GREEN)
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 0, 255) ,
            2
        )

        cv2.putText(
            image,
            "GT Polyp",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255)  ,
            1
        )

    # Save output
    cv2.imwrite(os.path.join(OUT_DIR, image_name), image)

    print(f"✅ Ground truth bbox drawn: {image_name}")

print("\n🎉 All ground-truth bounding boxes generated successfully!")
