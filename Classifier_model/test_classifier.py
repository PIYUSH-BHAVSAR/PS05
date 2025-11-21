import os
import cv2
import torch
import shutil
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b0
import torch.nn as nn

import pandas as pd

# ====================== CONFIG ======================
TEST_FOLDER = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\Classifier_model\test"
MODEL_PATH = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\Classifier_model\best_classifier.pth"
OUTPUT_FOLDER = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\Classifier_model\sorted_output"

IMG_SIZE = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Printed", "Handwritten"]

# Create output folders
os.makedirs(os.path.join(OUTPUT_FOLDER, "Printed"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "Handwritten"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "Low_Confidence"), exist_ok=True)

# ====================== TRANSFORMS ======================
test_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ====================== LOAD MODEL ======================
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Classifier loaded successfully!")


# ====================== PREDICT FUNCTION ======================
def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = test_tfms(image=img)["image"]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]

        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    return pred_class, confidence


# ====================== PROCESS FOLDER ======================
results = []

LOW_CONF_THRESHOLD = 0.70   # move uncertain samples here

image_files = [f for f in os.listdir(TEST_FOLDER)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for file in tqdm(image_files):
    img_path = os.path.join(TEST_FOLDER, file)

    pred_class, confidence = predict_image(img_path)

    if pred_class is None:
        continue

    class_name = CLASS_NAMES[pred_class]

    # Decide target folder
    if confidence < LOW_CONF_THRESHOLD:
        target_dir = os.path.join(OUTPUT_FOLDER, "Low_Confidence")
    else:
        target_dir = os.path.join(OUTPUT_FOLDER, class_name)

    shutil.copy(img_path, os.path.join(target_dir, file))

    results.append({
        "filename": file,
        "predicted_class": class_name,
        "confidence": round(confidence, 4)
    })


# ====================== SAVE REPORT ======================
df = pd.DataFrame(results)
df.to_csv("mixed_test_predictions.csv", index=False)

print("\n🎉 SEPARATION COMPLETE!")
print(f"Results saved to : mixed_test_predictions.csv")
print(f"Sorted images saved to : {OUTPUT_FOLDER}")
