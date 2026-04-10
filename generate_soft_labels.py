import os
import cv2
import torch
from ram_model import RAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RAM().to(device)
model.load_state_dict(torch.load("ram_model.pth", map_location=device))
model.eval()

bbox_folder = "unlabelled/bbox"
output_folder = "unlabelled/softlabels"

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(bbox_folder):

    if not file_name.endswith(".png"):
        continue

    path = os.path.join(bbox_folder, file_name)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (128,128))
    img = torch.tensor(img/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img).item()

    score = max(0, min(1, score))

    out_path = os.path.join(output_folder, file_name.replace(".png",".txt"))

    with open(out_path, "w") as f:
        f.write(str(score))

    print(f"{file_name} -> {score:.4f}")