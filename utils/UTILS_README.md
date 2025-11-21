# Utility Scripts Documentation

This folder contains utility scripts for dataset analysis, visualization, and performance benchmarking of the Multilingual Document Understanding system.

## Overview

The utilities help with:
- Dataset quality assurance and validation
- Visual debugging of annotations
- Performance profiling and resource monitoring
- Category distribution analysis

---

## Scripts

### 1. findcatzero.py

**Purpose:** Analyzes dataset annotations to count category distributions and identify missing or underrepresented categories.

**What it does:**
- Recursively scans a folder for JSON annotation files
- Counts the number of annotations per category ID
- Identifies categories with zero instances (useful for detecting data imbalance)
- Provides a summary report of category distribution

**Use Cases:**
- Validate dataset balance before training
- Identify missing categories in your dataset
- Quality assurance for annotation completeness
- Debug annotation issues

**How to use:**
```python
python utils/findcatzero.py
```

**Configuration:**
Edit the `folder` variable in the script to point to your dataset directory:
```python
folder = r"path/to/your/dataset/folder"
```

**Output Example:**
```
Scanning folder: D:\projects\...\train

========== CATEGORY SUMMARY ==========
Category 1: 1250 annotations
Category 2: 890 annotations
Category 3: 456 annotations
Category 4: 234 annotations
Category 5: 678 annotations

Total JSON files scanned: 500
Total files scanned (all types): 1000
=====================================
```

**Key Features:**
- Handles nested folder structures
- Error handling for corrupted JSON files
- Counts total files vs JSON files
- Sorted category output for easy reading

---

### 2. get_annoted_image.py

**Purpose:** Visualizes YOLO format annotations overlaid on images for visual verification.

**What it does:**
- Reads an image and its corresponding YOLO label file
- Converts YOLO normalized coordinates to pixel coordinates
- Draws bounding boxes on the image with class labels
- Displays the annotated image using matplotlib

**Use Cases:**
- Verify annotation accuracy
- Debug coordinate transformation issues
- Visual quality check of training data
- Create sample images for documentation

**How to use:**
```python
python utils/get_annoted_image.py
```

**Configuration:**
Edit the paths in the script:
```python
image_path = r"path/to/image.png"
label_path = r"path/to/label.txt"
```

**Features:**
- Color-coded bounding boxes by class
- Class ID labels on each box
- Automatic coordinate conversion (YOLO → pixel)
- High-resolution display with matplotlib

**Color Scheme:**
- Class 0: Green boxes
- Other classes: Red boxes

**YOLO Format Reference:**
Each line in the label file: `class_id center_x center_y width height`
- All coordinates are normalized (0-1 range)
- center_x, center_y: box center coordinates
- width, height: box dimensions

---

### 3. resource_calculation.py

**Purpose:** Comprehensive performance benchmarking and resource profiling for YOLO models.

**What it does:**
- Measures model size and parameter count
- Calculates computational complexity (FLOPs)
- Monitors CPU RAM usage during inference
- Tracks GPU VRAM allocation and peak usage
- Measures inference time and throughput (FPS)
- Provides detailed resource utilization report

**Use Cases:**
- Optimize model deployment
- Plan hardware requirements
- Compare different model variants
- Identify performance bottlenecks
- Validate resource constraints for production

**How to use:**
```python
python utils/resource_calculation.py
```

**Configuration:**
Edit these paths in the script:
```python
MODEL_PATH = r"runs/detect/train5/weights/best.pt"
TEST_IMAGE = r"path/to/test/image.png"
```

**Output Example:**
```
🔵 Loading model...

================= MODEL INFORMATION =================
🔹 Total Parameters: 25,902,640
🔹 Approx FLOPs: 51.81 GFLOPs
🔹 Model file size: 49.73 MB
=====================================================

🔵 Warming up model...
🔵 Measuring resource usage...

================= RESOURCE UTILIZATION =================
🧠 CPU RAM Used: 0.523 GB
🎮 GPU VRAM Used (current): 1.234 GB
🔥 GPU VRAM Peak: 2.456 GB

⚡ Average Inference Time: 0.0234 sec/image
🚀 Throughput (FPS): 42.74 images/second
========================================================
```

**Metrics Explained:**

**Model Information:**
- **Total Parameters:** Number of trainable weights in the model
- **FLOPs:** Floating-point operations (computational complexity)
- **Model file size:** Disk space required for the model

**Resource Utilization:**
- **CPU RAM Used:** Memory consumed during inference
- **GPU VRAM Used:** Current GPU memory allocation
- **GPU VRAM Peak:** Maximum GPU memory used (important for batch processing)
- **Average Inference Time:** Time per image (lower is better)
- **Throughput (FPS):** Images processed per second (higher is better)

**Features:**
- Automatic GPU detection and fallback to CPU
- Model warm-up to ensure accurate measurements
- Multiple iterations for reliable averages
- Compatible with all YOLO versions
- Error handling for different YOLO implementations

**Performance Optimization Tips:**
- If GPU VRAM Peak is high, reduce batch size or image size
- If FPS is low, consider using a smaller model variant
- Monitor CPU RAM for memory leaks during long runs

---

## Common Workflows

### Dataset Validation Workflow

1. **Check category distribution:**
   ```bash
   python utils/findcatzero.py
   ```
   - Ensure all categories have sufficient samples
   - Identify underrepresented classes

2. **Visual verification:**
   ```bash
   python utils/get_annoted_image.py
   ```
   - Spot-check random samples
   - Verify bounding box accuracy

### Pre-Deployment Workflow

1. **Benchmark performance:**
   ```bash
   python utils/resource_calculation.py
   ```
   - Measure inference speed
   - Check resource requirements

2. **Compare models:**
   - Run benchmark on different model variants
   - Choose optimal model for your hardware

### Debugging Workflow

1. **Annotation issues:**
   - Use `get_annoted_image.py` to visualize problematic images
   - Check if boxes align correctly with objects

2. **Performance issues:**
   - Use `resource_calculation.py` to identify bottlenecks
   - Monitor GPU/CPU usage patterns

---

## Requirements

All utility scripts require:
```
opencv-python-headless
matplotlib
psutil
torch
ultralytics
numpy
```

Install with:
```bash
pip install -r ../requirements.txt
```

---

## Tips and Best Practices

### For findcatzero.py:
- Run on both train and validation sets separately
- Aim for balanced category distribution (within 2x ratio)
- Categories with <50 samples may cause poor model performance

### For get_annoted_image.py:
- Check multiple random samples, not just one
- Verify boxes don't extend beyond image boundaries
- Ensure class IDs match your dataset.yaml configuration

### For resource_calculation.py:
- Run on representative test images (typical size and complexity)
- Test with different image sizes (imgsz parameter)
- Run multiple times to account for system variability
- Close other GPU-intensive applications for accurate measurements

---

## Troubleshooting

**findcatzero.py:**
- **Error: "No JSON files found"** → Check folder path is correct
- **Error: "JSON decode error"** → Some annotation files may be corrupted

**get_annoted_image.py:**
- **Image not displaying** → Check if running in environment with display support
- **Boxes look wrong** → Verify label file format matches YOLO standard

**resource_calculation.py:**
- **CUDA out of memory** → Reduce image size or use CPU mode
- **Model not found** → Verify MODEL_PATH points to valid .pt file
- **Low FPS** → Normal for first run; check after warm-up iterations

---

## Contributing

When adding new utilities:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include usage examples in this README
4. Handle errors gracefully
5. Support both Windows and Linux paths

---

## Future Enhancements

Potential additions:
- Batch processing for get_annoted_image.py
- Export category statistics to CSV
- Automated dataset splitting recommendations
- Memory profiling for training process
- Comparative benchmarking across multiple models
