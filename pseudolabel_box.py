import os
import cv2

# ================= PATHS =================
IMG_DIR = "unlabelled/img"
LABEL_DIR = "unlabelled/pseudo_labels"
OUT_DIR = "unlabelled/pseudolabel_visuals"

os.makedirs(OUT_DIR, exist_ok=True)

# ================= IMAGE SIZE =================
IMG_WIDTH = 384
IMG_HEIGHT = 288

# ================= MAIN LOOP =================
for label_file in os.listdir(LABEL_DIR):

    if not label_file.endswith(".txt"):
        continue

    image_name = label_file.replace(".txt", ".png")
    image_path = os.path.join(IMG_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Image missing: {image_name}")
        continue

    # Load image
    image = cv2.imread(image_path)

    label_path = os.path.join(LABEL_DIR, label_file)

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"Invalid label format in {label_file}")
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # ================= YOLO → PIXEL =================
        x_center *= IMG_WIDTH
        y_center *= IMG_HEIGHT
        width *= IMG_WIDTH
        height *= IMG_HEIGHT

        x_min = int(x_center - width / 2)
        x_max = int(x_center + width / 2)
        y_min = int(y_center - height / 2)
        y_max = int(y_center + height / 2)

        # ================= DRAW BOX =================
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),   # BLUE for pseudo-label
            2
        )

        cv2.putText(
            image,
            "Pseudo Polyp",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    # ================= SAVE OUTPUT =================
    cv2.imwrite(os.path.join(OUT_DIR, image_name), image)

    print(f"Pseudo-label bbox drawn: {image_name}")

print("\nAll pseudo-label visualizations generated successfully!")