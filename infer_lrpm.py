import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from lrpm_ham_model import LRPM_HAM_Detector

# --------------------
# Paths
# --------------------
UNLABELLED_IMG_DIR = "unlabelled/img"
OUTPUT_DIR = "unlabelled/bbox"
SOFTMAX_DIR = "unlabelled/softmax"
MODEL_PATH = "lrpm_ham_detector.pth"

# Create folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SOFTMAX_DIR, exist_ok=True)

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# Load model
# --------------------
model = LRPM_HAM_Detector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------
# Transform
# --------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

CONF_THRESHOLD = 0.5

# --------------------
# Inference Loop
# --------------------
for img_name in os.listdir(UNLABELLED_IMG_DIR):

    if not img_name.endswith(".png"):
        continue

    print(f"Processing: {img_name}")

    img_path = os.path.join(UNLABELLED_IMG_DIR, img_name)

    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = transform(image_rgb).to(device)

    # --------------------
    # Model inference
    # --------------------
    with torch.no_grad():
        outputs = model([img_tensor])

    output = outputs[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    # --------------------
    # Draw bounding boxes (only strong detections)
    # --------------------
    for box, score, label in zip(boxes, scores, labels):

        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            f"Polyp {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --------------------
    # Softmax (ONE value per image)
    # --------------------
    if len(scores) == 0:
        final_softmax = "N/A"
    else:
        max_score = float(np.max(scores))

        if max_score < CONF_THRESHOLD:
            final_softmax = 0
        else:
            final_softmax = round(max_score, 2)

    # --------------------
    # Save softmax file
    # --------------------
    txt_name = img_name.replace(".png", ".txt")
    txt_path = os.path.join(SOFTMAX_DIR, txt_name)

    with open(txt_path, "w") as f:
        f.write(f"{final_softmax}\n")

    # --------------------
    # Save output image
    # --------------------
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, image_bgr)

    # print(f"Saved image → {out_path}")
    print(f"Saved softmax → {txt_path} (Value: {final_softmax})")

# --------------------
print("\nInference completed successfully!")