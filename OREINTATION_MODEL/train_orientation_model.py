import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\OREINTATION_MODEL\orientation_dataset"
IMG_SIZE = 256
BATCH_SIZE = 32       # works on RTX 3050 (6GB)
EPOCHS = 12           # 10–15 gives best accuracy
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔥 Using device: {device}\n")


# ============================================================
# DATASET CLASS
# ============================================================
class OrientationDataset(Dataset):
    def __init__(self, root):
        self.paths = []
        self.labels = []

        for cls in ["0", "1", "2", "3"]:
            folder = os.path.join(root, cls)
            files = glob(os.path.join(folder, "*.jpg"))
            self.paths += files
            self.labels += [int(cls)] * len(files)

        print(f"🔢 Total samples found: {len(self.paths)}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # HWC → CHW, normalize 0–1
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# ============================================================
# MODEL DEFINITION (FAST SMALL CNN)
# ============================================================
class OrientationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 256 → 128

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 128 → 64

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 64 → 32

            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)   # 4 classes
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAIN LOADER
# ============================================================
dataset = OrientationDataset(DATASET_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ============================================================
# INITIALIZE MODEL + LOSS + OPTIMIZER
# ============================================================
model = OrientationNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ============================================================
# TRAINING LOOP
# ============================================================
print("\n🚀 Starting training...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0
    correct = 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"⭐ Epoch {epoch}: Accuracy = {acc:.4f}")

# ============================================================
# SAVE MODEL
# ============================================================
torch.save(model.state_dict(), "orientation_model.pth")
print("\n🎉 Training complete!")
print("📁 Model saved as: orientation_model.pth\n")
