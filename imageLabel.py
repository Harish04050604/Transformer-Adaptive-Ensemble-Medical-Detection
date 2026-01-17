import os
import random
import shutil

# Paths
IMG_DIR = "img"
ANN_DIR = "ann"

LABELLED_IMG_DIR = "labelled/img"
LABELLED_ANN_DIR = "labelled/ann"

os.makedirs(LABELLED_IMG_DIR, exist_ok=True)
os.makedirs(LABELLED_ANN_DIR, exist_ok=True)

# Get image filenames
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]

# Select 122 images
labelled_images = random.sample(image_files, 122)

copied = 0

for img_name in labelled_images:
    ann_name = img_name + ".json"   # ✅ FIX HERE

    img_path = os.path.join(IMG_DIR, img_name)
    ann_path = os.path.join(ANN_DIR, ann_name)

    if not os.path.exists(ann_path):
        print(f"❌ Annotation missing for {img_name}")
        continue

    shutil.copy(img_path, os.path.join(LABELLED_IMG_DIR, img_name))
    shutil.copy(ann_path, os.path.join(LABELLED_ANN_DIR, ann_name))
    copied += 1

print(f"✅ {copied} labelled images and annotations selected successfully.")
