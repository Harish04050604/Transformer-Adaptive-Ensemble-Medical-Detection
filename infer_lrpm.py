import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from lrpm_ham_model import LRPM_HAM_Detector

UNLABELLED_IMG_DIR = "unlabelled/img"
OUTPUT_DIR = "unlabelled/bbox"
MODEL_PATH = "lrpm_ham_detector.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LRPM_HAM_Detector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

CONF_THRESHOLD = 0.5  

for img_name in os.listdir(UNLABELLED_IMG_DIR):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(UNLABELLED_IMG_DIR, img_name)
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = transform(image_rgb).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    output = outputs[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

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

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, image_bgr)

    print(f"Processed: {img_name}")

print("Inference completed. Bounding boxes saved.")
