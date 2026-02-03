import os

BBOX_DIR = "labelled/bboxes"
YOLO_DIR = "labelled/yolo"

IMG_WIDTH = 384
IMG_HEIGHT = 288

os.makedirs(YOLO_DIR, exist_ok=True)

for bbox_file in os.listdir(BBOX_DIR):
    if not bbox_file.endswith(".txt"):
        continue

    bbox_path = os.path.join(BBOX_DIR, bbox_file)
    yolo_path = os.path.join(YOLO_DIR, bbox_file)

    with open(bbox_path, "r") as f:
        lines = f.readlines()

    yolo_lines = []

    for line in lines:
        parts = line.strip().split()

        class_name = parts[0]
        x_min = float(parts[1])
        y_min = float(parts[2])
        x_max = float(parts[3])
        y_max = float(parts[4])

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        x_center /= IMG_WIDTH
        y_center /= IMG_HEIGHT
        width /= IMG_WIDTH
        height /= IMG_HEIGHT

        class_id = 0

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        )

    with open(yolo_path, "w") as f:
        f.writelines(yolo_lines)

print("All bounding boxes converted to YOLO format successfully.")
