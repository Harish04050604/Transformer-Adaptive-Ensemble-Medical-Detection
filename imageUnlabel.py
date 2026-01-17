import os
import shutil

ALL_IMG_DIR = "img"
LABELLED_IMG_DIR = "labelled/img"
UNLABELLED_IMG_DIR = "unlabelled/img"

os.makedirs(UNLABELLED_IMG_DIR, exist_ok=True)

labelled_set = set(os.listdir(LABELLED_IMG_DIR))

for img_name in os.listdir(ALL_IMG_DIR):
    if img_name.endswith(".png") and img_name not in labelled_set:
        shutil.copy(
            os.path.join(ALL_IMG_DIR, img_name),
            os.path.join(UNLABELLED_IMG_DIR, img_name)
        )

print("Unlabelled images prepared.")
