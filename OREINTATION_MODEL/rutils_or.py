"""
deskew_predict_export.py
---------------------------------
Orientation → Deskew → YOLO Pipeline
also includes:
  🔹 Resource Usage Benchmark (CPU/GPU)
  🔹 Full inference time profiling
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Orientation model imports
import torch
from torchvision import transforms
import psutil
import time


# ------------------------- CONFIGURATION --------------------------
MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\runs\detect\train5\weights\best.pt"
SOURCE_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\images"
OUT_JSON_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\stage1_submission_jsons"
OUT_IMG_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\dataset\deskewed_predicted_images"

ORIENTATION_MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\OREINTATION_MODEL\orientation_model.pth"

CONF_THRESH = 0.25
IMG_SIZE = 960
ORI_IMG_SIZE = 256  # MUST match training
# ------------------------------------------------------------------

os.makedirs(OUT_JSON_DIR, exist_ok=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)


# --------------------- LOAD YOLO MODEL ---------------------
model = YOLO(MODEL_PATH)
print(f"✅ Loaded YOLO model from: {MODEL_PATH}")


# ================================================================
# >>> ORIENTATION MODEL (EXACT TRAIN ARCHITECTURE)
# ================================================================
class OrientationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(

            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),

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


# =====================================================================
# >>> RESOURCE USAGE INITIAL SNAPSHOT
# =====================================================================
process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gpu_before = torch.cuda.memory_allocated()
else:
    gpu_before = 0


# Benchmark timing
per_image_times = []


for idx, img_name in enumerate(image_files, 1):

    img_path = os.path.join(SOURCE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ Skipped unreadable file: {img_name}")
        continue

    start_time = time.time()

    # STEP 1: ORIENTATION FIX
    upright, rot_angle = fix_orientation(img)

    # STEP 2: DESKEW
    deskewed, tilt_angle = deskew_image(upright)

    # STEP 3: YOLO INFERENCE
    results = model.predict(
        deskewed,
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        save=False,
        verbose=False
    )

    # JSON formatting
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

    # Save JSON
    out_json_path = os.path.join(OUT_JSON_DIR, Path(img_name).stem + ".json")
    with open(out_json_path, "w") as jf:
        json.dump({"file_name": img_name, "annotations": annotations}, jf, indent=2)

    # Save image
    annotated = results[0].plot()
    out_img_path = os.path.join(OUT_IMG_DIR, img_name)
    cv2.imwrite(out_img_path, annotated)

    # Measure time
    per_image_times.append(time.time() - start_time)


# =====================================================================
# >>> RESOURCE USAGE AFTER PROCESSING
# =====================================================================
ram_after = process.memory_info().rss

if torch.cuda.is_available():
    gpu_after = torch.cuda.memory_allocated()
    gpu_peak = torch.cuda.max_memory_allocated()
else:
    gpu_after = gpu_peak = 0


def bytes_to_gb(x):
    return round(x / (1024**3), 3)



# =====================================================================
# >>> PRINT RESOURCE BENCHMARK SUMMARY
# =====================================================================
avg_time = sum(per_image_times) / len(per_image_times)
fps = 1 / avg_time if avg_time > 0 else 0

print("\n================= RESOURCE UTILIZATION =================")
print(f"🧠 CPU RAM Used: {bytes_to_gb(ram_after - ram_before)} GB")
print(f"🎮 GPU VRAM Used (current): {bytes_to_gb(gpu_after - gpu_before)} GB")
print(f"🔥 GPU VRAM Peak: {bytes_to_gb(gpu_peak)} GB")

print(f"\n⚡ Average Pipeline Time (Orientation+Deskew+YOLO): {avg_time:.4f} sec/image")
print(f"🚀 Pipeline Throughput: {fps:.2f} images/second")
print("========================================================\n")


print("✅ ALL DONE!")
print(f"📂 JSONs saved in: {OUT_JSON_DIR}")
print(f"🖼 Annotated images saved in: {OUT_IMG_DIR}")
