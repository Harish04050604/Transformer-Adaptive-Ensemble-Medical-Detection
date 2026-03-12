import os
import cv2
import numpy as np
from PIL import Image

INPUT_DIR = "unlabelled/bbox"
OUTPUT_DIR = "unlabelled/segment"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def segment_inside_box(image, bbox):

    x1,y1,x2,y2 = bbox

    crop = image[y1:y2, x1:x2]

    # convert to gray
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # threshold segmentation
    _, mask = cv2.threshold(gray,0,255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return mask


for img_name in os.listdir(INPUT_DIR):

    path = os.path.join(INPUT_DIR, img_name)

    image = cv2.imread(path)

    h,w,_ = image.shape

    # Example bounding box (replace with your predicted box)
    # x1,y1,x2,y2
    bbox = (50,120,200,280)

    mask = np.zeros((h,w), dtype=np.uint8)

    polyp_mask = segment_inside_box(image, bbox)

    x1,y1,x2,y2 = bbox

    mask[y1:y2,x1:x2] = polyp_mask

    save_path = os.path.join(OUTPUT_DIR,img_name)

    cv2.imwrite(save_path, mask)

    print("Saved:", img_name)