"""
deskew_export_final.py
------------------------------------------------
Pipeline:
Orientation → Deskew → Classifier →
→ Printed YOLO / Handwritten YOLO →
→ Save annotated images + Stage-1 JSON

Output:
<out>/
 ├── images/
 └── jsons/

Works on:
✔ Windows
✔ Linux
✔ Docker
✔ GPU / CPU
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
import argparse
import time

# Optional psutil
try:
    import psutil
except ImportError:
    psutil = None

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO


# =======================================================
# ARGUMENT PARSING
# =======================================================
parser = argparse.ArgumentParser(description="OCR Full Pipeline")

parser.add_argument("--source", type=str, default=None,
                    help="Input folder of images")

parser.add_argument("--out", type=str, default=None,
                    help="Output folder name or full path")

parser.add_argument("--no-images", action="store_true",
                    help="If set, do NOT save annotated images")

args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path(args.source).resolve() if args.source else SCRIPT_DIR

if args.out:
    OUT_PATH = Path(args.out)
    BASE_OUT = OUT_PATH if OUT_PATH.is_absolute() else SOURCE_DIR / OUT_PATH
else:
    BASE_OUT = SOURCE_DIR / "results"

BASE_OUT.mkdir(parents=True, exist_ok=True)

OUT_JSON_DIR = BASE_OUT / "jsons"
OUT_JSON_DIR.mkdir(exist_ok=True)

# Only create images folder if enabled
SAVE_IMAGES = not args.no_images
if SAVE_IMAGES:
    OUT_IMAGE_DIR = BASE_OUT / "images"
    OUT_IMAGE_DIR.mkdir(exist_ok=True)


# =======================================================
# MODEL PATHS (AUTO PORTABLE)
# =======================================================
BASE_MODEL = SCRIPT_DIR / "models"

ORIENTATION_MODEL_PATH = BASE_MODEL / "orientation_model.pth"
CLASSIFIER_MODEL_PATH = BASE_MODEL / "best_classifier.pth"
PRINTED_YOLO_PATH = BASE_MODEL / "printed" / "best.pt"
HANDWRITTEN_YOLO_PATH = BASE_MODEL / "handwritten" / "best.pt"

print("\n🔍 Model Paths Check:")
print("Orientation:", ORIENTATION_MODEL_PATH)
print("Classifier :", CLASSIFIER_MODEL_PATH)
print("Printed YOLO:", PRINTED_YOLO_PATH)
print("Handwritten YOLO:", HANDWRITTEN_YOLO_PATH)

for path in [ORIENTATION_MODEL_PATH, CLASSIFIER_MODEL_PATH,
             PRINTED_YOLO_PATH, HANDWRITTEN_YOLO_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing file: {path}")


# =======================================================
# SETTINGS
# =======================================================
CONF_THRESH = 0.25
IMG_SIZE = 960
ORI_IMG_SIZE = 256
CLS_THRESHOLD = 0.75

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Running on device: {device.upper()}")


# =======================================================
# LOAD YOLO MODELS
# =======================================================
print("\n🔄 Loading YOLO models...")
printed_yolo = YOLO(str(PRINTED_YOLO_PATH))
handwritten_yolo = YOLO(str(HANDWRITTEN_YOLO_PATH))
print("✅ YOLO models loaded successfully.")


# =======================================================
# ORIENTATION MODEL
# =======================================================
class OrientationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (ORI_IMG_SIZE // 8) * (ORI_IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)


orientation_model = OrientationNet().to(device)
orientation_model.load_state_dict(
    torch.load(ORIENTATION_MODEL_PATH, map_location=device),
    strict=False
)
orientation_model.eval()

ori_transform = transforms.Compose([transforms.ToTensor()])


# =======================================================
# CLASSIFIER MODEL
# =======================================================
classifier = efficientnet_b0(weights=None)
classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, 2)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier.to(device)
classifier.eval()

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])


# =======================================================
# FUNCTIONS
# =======================================================
def fix_orientation(img):
    try:
        resized = cv2.resize(img, (ORI_IMG_SIZE, ORI_IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = ori_transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            cls = orientation_model(inp).argmax(1).item()

        if cls == 1:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif cls == 2:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif cls == 3:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img
    except:
        return img


def deskew_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)

        thresh = cv2.threshold(inv, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50:
            return img

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        angle = -angle

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)

        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except:
        return img


def classify_image(img):
    try:
        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = cls_transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(classifier(tensor), dim=1)[0]

        return probs.argmax().item(), probs.max().item()
    except:
        return 0, 0.5


def run_yolo(img, model):
    try:
        results = model.predict(
            img,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            verbose=False
        )[0]

        boxes = results.boxes.xywh.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        annotations = []
        for (x, y, w, h), c, cf in zip(boxes, classes, confs):
            if cf < CONF_THRESH:
                continue
            annotations.append({
                "bbox": [float(x-w/2), float(y-h/2), float(w), float(h)],
                "category_id": int(c + 1)
            })

        annotated_img = results.plot() if SAVE_IMAGES else None

        return annotations, annotated_img
    except:
        return [], img


def print_resource_usage(start_time):
    print("\n📊 RESOURCE USAGE")

    if psutil:
        print(f"CPU: {psutil.cpu_percent()}%")
        print(f"RAM: {psutil.virtual_memory().percent}%")

    if torch.cuda.is_available():
        print(f"GPU VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print(f"Total Time: {time.time() - start_time:.2f} sec")


# =======================================================
# MAIN PIPELINE
# =======================================================
start_time = time.time()

images = [x for x in SOURCE_DIR.iterdir()
          if x.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"\n📂 SOURCE: {SOURCE_DIR}")
print(f"📦 OUTPUT: {BASE_OUT}")
print(f"🖼 Found {len(images)} images")
print(f"📸 Image saving: {'ON' if SAVE_IMAGES else 'OFF'}\n")

for idx, img_path in enumerate(images, 1):
    print(f"[{idx}/{len(images)}] {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print("⚠ Skipped unreadable image")
        continue

    img = fix_orientation(img)
    img = deskew_image(img)

    cls, conf = classify_image(img)

    if cls == 1 and conf >= CLS_THRESHOLD:
        model = handwritten_yolo
        label = "HANDWRITTEN"
    else:
        model = printed_yolo
        label = "PRINTED"

    print(f"➡ {label}")

    annotation, annotated = run_yolo(img, model)

    with open(OUT_JSON_DIR / f"{img_path.stem}.json", "w") as f:
        json.dump({
            "file_name": img_path.name,
            "annotations": annotation
        }, f, indent=2)

    if SAVE_IMAGES:
        cv2.imwrite(str(OUT_IMAGE_DIR / img_path.name), annotated)

print("\n✅ PIPELINE FINISHED")
print_resource_usage(start_time)
