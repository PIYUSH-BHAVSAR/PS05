"""
Full Resource Usage Benchmark for YOLO Model (Compatible with all versions)
"""

import os
import cv2
import time
import psutil
import torch
from ultralytics import YOLO

MODEL_PATH = r"runs/detect/train5/weights/best.pt"
TEST_IMAGE = r"D:\projects\MULTILINGUAL_OCR\multilingual_doc_understanding\ovr_v2\stage1_dataset\images\train\doc_00000.png"  # change to any real image


def bytes_to_gb(x):
    return round(x / (1024 ** 3), 3)


# --------------------- LOAD MODEL ---------------------
print("\n🔵 Loading model...")
model = YOLO(MODEL_PATH)

print("\n================= MODEL INFORMATION =================")

# SAFE extraction of model details (works on all YOLO versions)
try:
    model.info()
except:
    print("⚠ model.info() not supported, skipping...")

# Manual params and FLOPs extraction (always works)
try:
    params = sum(p.numel() for p in model.model.parameters())
    print(f"🔹 Total Parameters: {params:,}")
except:
    print("⚠ Could not extract parameter count.")

# Manual FLOPs test
try:
    dummy = torch.zeros(1, 3, 640, 640)
    if torch.cuda.is_available():
        dummy = dummy.cuda()
        model.model.cuda()

    flops = sum(p.numel() for p in model.model.parameters()) * 2
    print(f"🔹 Approx FLOPs: {flops/1e9:.2f} GFLOPs")
except:
    print("⚠ Could not compute FLOPs.")

# Model size
try:
    print(f"🔹 Model file size: {round(os.path.getsize(MODEL_PATH)/1024/1024, 2)} MB")
except:
    pass

print("=====================================================\n")

# Load test image
img = cv2.imread(TEST_IMAGE)
if img is None:
    raise ValueError("❌ TEST_IMAGE path is wrong.")

# Warm-up
print("🔵 Warming up model...")
for _ in range(3):
    model.predict(img, imgsz=960)


# --------------------- MEASURE RESOURCES ---------------------
print("\n🔵 Measuring resource usage...\n")

process = psutil.Process(os.getpid())
ram_before = process.memory_info().rss

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gpu_before = torch.cuda.memory_allocated()
else:
    gpu_before = 0

ITER = 10
start = time.time()
for _ in range(ITER):
    model.predict(img, imgsz=960)
end = time.time()

avg_time = (end - start) / ITER
fps = 1 / avg_time

ram_after = process.memory_info().rss

if torch.cuda.is_available():
    gpu_after = torch.cuda.memory_allocated()
    gpu_peak = torch.cuda.max_memory_allocated()
else:
    gpu_after = 0
    gpu_peak = 0

# --------------------- PRINT RESULTS ---------------------
print("\n================= RESOURCE UTILIZATION =================")
print(f"🧠 CPU RAM Used: {bytes_to_gb(ram_after - ram_before)} GB")
print(f"🎮 GPU VRAM Used (current): {bytes_to_gb(gpu_after - gpu_before)} GB")
print(f"🔥 GPU VRAM Peak: {bytes_to_gb(gpu_peak)} GB")

print(f"\n⚡ Average Inference Time: {avg_time:.4f} sec/image")
print(f"🚀 Throughput (FPS): {fps:.2f} images/second")
print("========================================================\n")
