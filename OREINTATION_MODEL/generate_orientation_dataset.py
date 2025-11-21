import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\lang_data"      # <-- ONLY CHANGE THIS
OUTPUT_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\OREINTATION_MODEL\orientation_dataset"
IMG_SIZE = 256
# ============================================================


# ------------------------- DESKEW FUNCTION ------------------------
def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle -= 90

    angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


# ============================================================
# ENSURE DATASET FOLDERS EXIST
# ============================================================
for cls in ["0", "1", "2", "3"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ============================================================
# FIND EXISTING COUNT AND CONTINUE SAFELY
# ============================================================
existing_files = list(Path(OUTPUT_DIR).rglob("*.jpg"))
existing_samples_count = len(existing_files) // 4  # each sample has 4 rotations

idx = existing_samples_count
print(f"🔢 Existing samples: {existing_samples_count}")
print(f"➡ Continuing from index: {idx}\n")


# ============================================================
# LOAD NEW IMAGES ONLY
# ============================================================
image_files = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_files += list(Path(SOURCE_DIR).rglob(ext))

print(f"🟦 Found {len(image_files)} NEW raw images in SOURCE_DIR")

# ============================================================
# PROCESS NEW IMAGES (APPEND MODE)
# ============================================================
for img_path in tqdm(image_files, desc="Processing new images"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Step 1 — Deskew
    deskewed, _ = deskew_image(img)
    deskewed = cv2.resize(deskewed, (IMG_SIZE, IMG_SIZE))

    # Save 0° upright
    cv2.imwrite(f"{OUTPUT_DIR}/0/{idx}_0.jpg", deskewed)

    # Save 90° rotated
    rot90 = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{OUTPUT_DIR}/1/{idx}_1.jpg", rot90)

    # Save 180° rotated
    rot180 = cv2.rotate(deskewed, cv2.ROTATE_180)
    cv2.imwrite(f"{OUTPUT_DIR}/2/{idx}_2.jpg", rot180)

    # Save 270° rotated
    rot270 = cv2.rotate(deskewed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f"{OUTPUT_DIR}/3/{idx}_3.jpg", rot270)

    idx += 1

print("\n===================================================")
print("🎉 NEW DATA APPENDED SUCCESSFULLY!")
print(f"📁 Orientation dataset stored at: {OUTPUT_DIR}")
print(f"🔢 Total samples now: {idx}")
print("===================================================\n")
