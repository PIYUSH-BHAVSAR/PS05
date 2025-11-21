import os
import shutil
from collections import Counter

# ==============================
# EDIT THESE PATHS
# ==============================
IMG_DIR = "handwritten_yolo/images/train"
LBL_DIR = "handwritten_yolo/labels/train"
OUT_IMG_DIR = "handwritten_yolo_aug/images/train"
OUT_LBL_DIR = "handwritten_yolo_aug/labels/train"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# ==============================
# CLASS DUPLICATION FACTORS
# Change if needed
# ==============================
dup_factors = {
    0: 1,   # Text
    1: 3,   # Title
    2: 5,   # List
    3: 7,   # Table
    4: 10   # Figure
}

# =====================================
# Function to get class ids from label
# =====================================
def get_classes_from_label(label_path):
    classes = set()
    with open(label_path, "r") as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            classes.add(cls_id)
    return classes

# =====================================
# Process all images
# =====================================
count = 0
class_counter = Counter()

for img_file in os.listdir(IMG_DIR):
    if not img_file.endswith((".jpg", ".png", ".jpeg")):
        continue

    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + ".txt"

    img_path = os.path.join(IMG_DIR, img_file)
    label_path = os.path.join(LBL_DIR, label_file)

    if not os.path.exists(label_path):
        continue

    classes = get_classes_from_label(label_path)

    # find max duplication factor based on most rare class in image
    factor = max([dup_factors.get(cls, 1) for cls in classes])

    for i in range(factor):
        new_img_name = f"{base_name}_dup{i}.jpg"
        new_lbl_name = f"{base_name}_dup{i}.txt"

        shutil.copy(img_path, os.path.join(OUT_IMG_DIR, new_img_name))
        shutil.copy(label_path, os.path.join(OUT_LBL_DIR, new_lbl_name))

        for cls in classes:
            class_counter[cls] += 1

        count += 1

print("\n✅ DATASET DUPLICATION COMPLETE")
print(f"Total augmented images: {count}")
print("Class distribution after duplication:")
print(class_counter)
