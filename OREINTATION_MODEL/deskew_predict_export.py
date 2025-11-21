"""
deskew_predict_export.py
---------------------------------
Deskews input document images, runs YOLOv8 inference,
and saves both:
  1️⃣ Upright + deskewed + annotated images
  2️⃣ JSONs in official Stage-1 submission format

Author: Piyush Bhavsar
Modified by: ChatGPT (Orientation + Deskew version)
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# >>> ADDED FOR ORIENTATION MODEL
import torch
from torchvision import transforms


# ------------------------- CONFIGURATION --------------------------
MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\runs\detect\train5\weights\best.pt"
SOURCE_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\images"
OUT_JSON_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\stage1_submission_jsons"
OUT_IMG_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\deskewed_predicted_images"

ORIENTATION_MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\OREINTATION_MODEL\orientation_model.pth"

CONF_THRESH = 0.25
IMG_SIZE = 960

# MUST MATCH TRAINING IMAGE SIZE
ORI_IMG_SIZE = 256
# ------------------------------------------------------------------

os.makedirs(OUT_JSON_DIR, exist_ok=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)
print(f"✅ Loaded YOLO model from: {MODEL_PATH}")


# ================================================================
# >>> ORIENTATION MODEL DEFINITION (EXACT TRAIN ARCHITECTURE)
# ================================================================
class OrientationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(

            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 256 → 128

            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 128 → 64

            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 64 → 32

            torch.nn.Flatten(),

            # IMPORTANT: SAME AS TRAINING — DO NOT CHANGE
            torch.nn.Linear(128 * (ORI_IMG_SIZE // 8) * (ORI_IMG_SIZE // 8), 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

orientation_model = OrientationNet().to(device)
orientation_model.load_state_dict(torch.load(ORIENTATION_MODEL_PATH, map_location=device))
orientation_model.eval()

transform = transforms.Compose([transforms.ToTensor()])

def fix_orientation(img):
    """
    Predicts orientation class (0/90/180/270) and rotates to upright.
    """
    inp = cv2.resize(img, (ORI_IMG_SIZE, ORI_IMG_SIZE))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = transform(inp).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = orientation_model(inp)
        cls = pred.argmax(1).item()

    if cls == 0:
        return img, 0
    elif cls == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90
    elif cls == 2:
        return cv2.rotate(img, cv2.ROTATE_180), 180
    else:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270


print("✅ Orientation model loaded.")


def deskew_image(img):
    # ---- Downscale for fast angle calculation ----
    h, w = img.shape[:2]
    new_h = 800
    scale_ratio = new_h / h
    small = cv2.resize(img, (int(w * scale_ratio), new_h), interpolation=cv2.INTER_LINEAR)

    # ---- Deskew on small image ----
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
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

    # ---- Rotate ORIGINAL FULL-RES IMAGE ----
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle




# ---------------------- MAIN INFERENCE LOOP ----------------------
image_files = [f for f in os.listdir(SOURCE_DIR)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"🧠 Found {len(image_files)} images.")
print("🚀 Starting ORIENTATION → DESKEW → YOLO pipeline ...")


for idx, img_name in enumerate(image_files, 1):
    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipped unreadable file: {img_name}")
        continue


    # ----------------- STEP 1: ORIENTATION FIX -----------------
    upright, rot_angle = fix_orientation(img)
    print(f"[{idx}] Orientation corrected by {rot_angle}°")


    # ----------------- STEP 2: DESKEW -------------------------
    deskewed, tilt_angle = deskew_image(upright)
    print(f"    Deskew tilt = {tilt_angle:.2f}°")


    # ----------------- STEP 3: YOLO PREDICTION ----------------
    results = model.predict(
        deskewed,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=False,
        verbose=False
    )

    boxes = results[0].boxes.xywh.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    annotations = []
    for (x, y, w, h), cid, cf in zip(boxes, classes, confs):
        if cf < CONF_THRESH:
            continue
        annotations.append({
            "bbox": [float(x - w / 2), float(y - h / 2), float(w), float(h)],
            "category_id": int(cid + 1)
        })

    out_json_path = os.path.join(OUT_JSON_DIR, Path(img_name).stem + ".json")
    with open(out_json_path, "w") as jf:
        json.dump({"file_name": img_name, "annotations": annotations}, jf, indent=2)

    annotated = results[0].plot()
    out_img_path = os.path.join(OUT_IMG_DIR, img_name)
    cv2.imwrite(out_img_path, annotated)


print("\n✅ ALL DONE!")
print(f"📂 JSONs saved in: {OUT_JSON_DIR}")
print(f"🖼 Annotated images saved in: {OUT_IMG_DIR}")
