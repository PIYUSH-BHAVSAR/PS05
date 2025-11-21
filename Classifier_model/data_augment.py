import os
import cv2
import albumentations as A
from tqdm import tqdm
from datetime import datetime

# =============================
# CONFIG
# =============================

INPUT_DIR = "handwritten_data_deskew_images"     # Folder with 200 handwritten images
OUTPUT_DIR = "handwritten_augmented"   # Output folder
AUG_PER_IMAGE = 20                     # <-- Increase if you want more (20 = 4000 images)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# AUGMENTATION PIPELINE
# =============================

transform = A.Compose([
    A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, value=255, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.GaussianBlur(blur_limit=(3,5), p=0.5),
    A.GaussNoise(var_limit=(10,50), p=0.7),
    A.ElasticTransform(alpha=50, sigma=10, alpha_affine=10, p=0.3),
    A.Perspective(scale=(0.02,0.06), p=0.6),
    A.RandomShadow(p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
    A.Affine(shear=10, p=0.4),
    A.RandomScale(scale_limit=0.2, p=0.5),
], p=1.0)

# =============================
# PROCESS IMAGES
# =============================

images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print(f"Found {len(images)} handwritten images.")
print(f"Generating {AUG_PER_IMAGE} augmentations per image …")

counter = 0

for img_name in tqdm(images):
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]

    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=img)["image"]
        out_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), augmented)
        counter += 1

print(f"\nDONE! Generated {counter} augmented handwritten samples.")
print(f"All saved in: {OUTPUT_DIR}")
