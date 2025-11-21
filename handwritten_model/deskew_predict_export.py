"""
deskew_predict_export.py
---------------------------------
Deskews input document images, runs YOLOv8 inference,
and saves both:
  1️⃣ Deskewed + annotated images (for visual verification)
  2️⃣ JSONs in official Stage-1 submission format

  
Author: Piyush Bhavsar
Modified & Fixed by: ChatGPT (Arjan's version)
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ------------------------- CONFIGURATION --------------------------
MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\handwritten_model\best.pt"  # your trained YOLO model
SOURCE_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\handwritten_model\test"  # folder with tilted images
OUT_JSON_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\handwritten_model\stage1_submission_jsons"  # where JSONs will go
OUT_IMG_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\handwritten_model\deskewed_predicted_images"  # annotated deskewed images for verification

CONF_THRESH = 0.25  # confidence threshold
IMG_SIZE = 960      # YOLO inference image size
# ------------------------------------------------------------------

os.makedirs(OUT_JSON_DIR, exist_ok=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")


# ------------------------- DESKEW FUNCTION ------------------------
def deskew_image(img):
    """
    Deskew the image by estimating the dominant text-line or document angle.
    Returns the deskewed image and the correction angle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert for text-on-white images
    gray_inv = cv2.bitwise_not(gray)

    # Threshold to binary
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Get coordinates of all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        # fallback: return original if no text content detected
        return img, 0.0

    # Compute angle using minAreaRect
    angle = cv2.minAreaRect(coords)[-1]

    # Normalize the angle to be between [-45, 45]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Negate to rotate in the correct direction
    angle = -angle

    # Rotate image around its center
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle


# ---------------------- MAIN INFERENCE LOOP ----------------------
image_files = [f for f in os.listdir(SOURCE_DIR)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"🧠 Found {len(image_files)} images in {SOURCE_DIR}")
print("🚀 Starting deskew + YOLO prediction ...")

for idx, img_name in enumerate(image_files, 1):
    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipped unreadable file: {img_name}")
        continue

    # Step 1: Deskew
    deskewed, angle = deskew_image(img)
    print(f"[{idx}/{len(image_files)}] Deskewed {img_name} by {angle:.2f}°")

    # Step 2: Run YOLO inference on deskewed image
    results = model.predict(
        deskewed,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=False,
        verbose=False
    )

    # Extract detections
    boxes = results[0].boxes.xywh.cpu().numpy()   # x_center, y_center, w, h
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    # Step 3: Prepare annotations in official JSON format
    annotations = []
    for (x, y, w, h), cid, cf in zip(boxes, classes, confs):
        if cf < CONF_THRESH:
            continue
        annotations.append({
            "bbox": [float(x - w / 2), float(y - h / 2), float(w), float(h)],
            "category_id": int(cid + 1)  # YOLO uses 0-based, dataset uses 1-based IDs
        })

    output_json = {
        "file_name": img_name,
        "annotations": annotations
    }

    # Step 4: Save JSON file
    out_json_path = os.path.join(OUT_JSON_DIR, Path(img_name).stem + ".json")
    with open(out_json_path, "w") as jf:
        json.dump(output_json, jf, indent=2)

    # Step 5: Save annotated deskewed image
    annotated = results[0].plot()  # returns image with bounding boxes
    out_img_path = os.path.join(OUT_IMG_DIR, img_name)
    cv2.imwrite(out_img_path, annotated)

print("\n✅ All images processed successfully!")
print(f"📂 JSONs saved to: {OUT_JSON_DIR}")
print(f"🖼 Annotated deskewed images saved to: {OUT_IMG_DIR}")
print("\n👉 You can now zip the folder for submission:")
print("   Compress-Archive -Path stage1_submission_jsons\\* -DestinationPath stage1_submission.zip") 