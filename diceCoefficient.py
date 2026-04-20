import os
import json
import base64
import zlib
import numpy as np
import cv2

ANN_DIR = "ann"
PSEUDO_DIR = "unlabelled/pseudo_labels"

IMG_WIDTH = 384
IMG_HEIGHT = 288


def decode_bitmap(bitmap_data, image_height, image_width, origin):
    compressed = base64.b64decode(bitmap_data)
    png_bytes = zlib.decompress(compressed)

    mask = cv2.imdecode(
        np.frombuffer(png_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    mask = (mask > 0).astype(np.uint8)

    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    x0, y0 = origin
    h, w = mask.shape

    full_mask[y0:y0+h, x0:x0+w] = mask
    return full_mask


def yolo_to_mask(yolo_file):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    if not os.path.exists(yolo_file):
        return mask

    with open(yolo_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(float, parts)

            xc *= IMG_WIDTH
            yc *= IMG_HEIGHT
            w *= IMG_WIDTH
            h *= IMG_HEIGHT

            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)

            # clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(IMG_WIDTH, x2), min(IMG_HEIGHT, y2)

            mask[y1:y2, x1:x2] = 1

    return mask


def dice_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-6)


# 🔥 Get only unlabelled files
pseudo_files = [f for f in os.listdir(PSEUDO_DIR) if f.endswith(".txt")]

dice_scores = []

if not pseudo_files:
    print("No pseudo label files found!")
    exit()

for txt_file in pseudo_files:
    base_name = txt_file.replace(".txt", "")
    json_file = base_name + ".png.json"

    ann_path = os.path.join(ANN_DIR, json_file)
    pred_path = os.path.join(PSEUDO_DIR, txt_file)

    if not os.path.exists(ann_path):
        print(f"{json_file} → skipped (no annotation)")
        continue

    with open(ann_path) as f:
        data = json.load(f)

    height = data["size"]["height"]
    width = data["size"]["width"]

    gt_mask = np.zeros((height, width), dtype=np.uint8)

    for obj in data["objects"]:
        bitmap = obj["bitmap"]

        mask = decode_bitmap(
            bitmap["data"],
            height,
            width,
            bitmap["origin"]
        )

        gt_mask |= mask

    pred_mask = yolo_to_mask(pred_path)

    dice = dice_coefficient(gt_mask, pred_mask)
    dice_scores.append(dice)

    print(f"{json_file} → Dice: {dice:.4f}")


#  Compute mean Dice
if dice_scores:
    mean_dice = np.mean(dice_scores)
    print("\n==============================")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print("==============================")
else:
    print("No valid Dice scores computed.")