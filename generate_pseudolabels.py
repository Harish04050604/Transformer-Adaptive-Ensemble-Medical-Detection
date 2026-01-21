import os
import cv2
import torch
from lrpm_ham_model import LRPM_HAM_Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LRPM_HAM_Detector().to(device)
model.load_state_dict(torch.load("lrpm_ham_detector.pth", map_location=device))
model.eval()

IMG_DIR = "unlabelled/img"
OUT_DIR = "unlabelled/bbox"
os.makedirs(OUT_DIR, exist_ok=True)

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)

    img_tensor = torch.tensor(img / 255.0, dtype=torch.float32)\
                        .permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]

    for i, box in enumerate(boxes):
        if scores[i] < 0.7:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img[y1:y2, x1:x2]

        out_name = f"{img_name}_{i}.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), crop)

print("Pseudo bounding boxes generated for unlabelled images.")
