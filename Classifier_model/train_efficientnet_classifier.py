import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ===================================================
# CONFIG
# ===================================================
PRINTED_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\data\deskew_images"
HANDWRITTEN_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\classifier_model\handwritten_augmented"

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# Windows-safe dataloader settings
NUM_WORKERS = 0
PIN_MEMORY = False

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = GradScaler("cuda")


# ===================================================
# ALBUMENTATIONS (fixed)
# ===================================================
printed_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
    A.GaussianBlur(blur_limit=5, p=0.2),
    A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.8),

    # Normalize for EfficientNet
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

handwritten_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.9),
    A.RandomBrightnessContrast(0.4, 0.4, p=0.6),

    A.GaussNoise(var_limit=(10, 50), p=0.6),
    A.Perspective(scale=(0.02, 0.06), p=0.5),
    A.ElasticTransform(alpha=45, sigma=10, p=0.4),

    # Normalize for EfficientNet
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ===================================================
# DATASET CLASS
# ===================================================
class PrintedHandwrittenDataset(Dataset):
    def __init__(self, printed_dir, handwritten_dir):
        self.samples = []

        for f in os.listdir(printed_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(printed_dir, f), 0))

        for f in os.listdir(handwritten_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(handwritten_dir, f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if label == 0:
            img = printed_tfms(image=img)["image"]
        else:
            img = handwritten_tfms(image=img)["image"]

        return img, label


# ===================================================
# MAIN TRAINING (Windows multiprocessing fix)
# ===================================================
if __name__ == "__main__":

    dataset = PrintedHandwrittenDataset(PRINTED_DIR, HANDWRITTEN_DIR)

    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))

    weights = 1.0 / class_counts.float()
    sample_weights = [weights[label] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Printed samples     :", class_counts[0].item())
    print("Handwritten samples :", class_counts[1].item())


    # ===================================================
    # MODEL INIT (TorchVision – stable & optimized)
    # ===================================================
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    # ===================================================
    # TRAIN LOOP
    # ===================================================
    for epoch in range(EPOCHS):
        correct = 0
        total = 0

        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=f"{correct / total:.4f}")

        acc = correct / total
        print(f"Epoch {epoch + 1} → Accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_classifier.pth")
            print("✔ Saved BEST model!")

    print("\n🔥 Training Complete!")
    print(f"💎 Best Accuracy Achieved: {best_acc * 100:.2f}%")
