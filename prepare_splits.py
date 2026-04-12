# prepare_splits.py

import os
import random

IMAGE_DIR = os.path.join("labelled", "img")

all_images = sorted(os.listdir(IMAGE_DIR))

random.seed(42)
random.shuffle(all_images)

split      = int(0.85 * len(all_images))
train_set  = all_images[:split]
val_set    = all_images[split:]

with open("train_split.txt", "w") as f:
    f.write("\n".join(train_set))

with open("val_split.txt", "w") as f:
    f.write("\n".join(val_set))

print(f"Total  : {len(all_images)}")
print(f"Train  : {len(train_set)}")
print(f"Val    : {len(val_set)}")
print("Splits saved: train_split.txt, val_split.txt")