# Training Steps - English Document Understanding Model

This guide provides step-by-step instructions for training a YOLOv8 model on English documents to detect and classify document elements (Text, Title, List, Table, Figure).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Data Preprocessing](#data-preprocessing)
4. [Dataset Conversion](#dataset-conversion)
5. [Training Configuration](#training-configuration)
6. [Model Training](#model-training)
7. [Training Monitoring](#training-monitoring)
8. [Model Evaluation](#model-evaluation)
9. [Inference and Export](#inference-and-export)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB+ VRAM (GTX 1070 or better)
- RAM: 16GB
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA RTX 3090 or A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 100GB+ SSD

### Software Requirements

1. **Python 3.8+**
2. **CUDA 11.8+** (for GPU acceleration)
3. **Required packages:**

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- ultralytics (YOLOv8)
- opencv-python-headless
- numpy
- tqdm

### Dataset Structure

Your raw English dataset should be organized as follows:

```
data/
└── english_Dataset/
    ├── doc_00001.png
    ├── doc_00001.json
    ├── doc_00002.png
    ├── doc_00002.json
    └── ...
```

**JSON Annotation Format:**
```json
{
  "file_name": "doc_00001.png",
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1
    }
  ]
}
```

**Category IDs:**
- 1: Text
- 2: Title
- 3: List
- 4: Table
- 5: Figure

---

## Dataset Preparation

### Step 1: Verify Dataset Quality

Before training, validate your dataset:

```bash
# Check category distribution
python ../utils/findcatzero.py
```

Edit the script to point to your dataset:
```python
folder = r"data/english_Dataset"
```

**What to look for:**
- All 5 categories should have annotations
- Balanced distribution (no category should have <10% of total)
- Minimum 500+ annotations per category recommended

### Step 2: Visual Inspection

Randomly inspect a few samples to ensure annotation quality:

```bash
python ../utils/get_annoted_image.py
```

**Check for:**
- Bounding boxes align correctly with elements
- No overlapping or duplicate annotations
- Boxes don't extend beyond image boundaries

---

## Data Preprocessing

### Step 3: Deskew Images (Recommended)

Deskewing corrects tilted/rotated documents, significantly improving model accuracy.

**Command:**
```bash
python deskew_and_transform.py --img_dir ../data/english_Dataset --ann_dir ../data/english_Dataset --out_img_dir ../data/deskewed_images --out_ann_dir ../data/deskewed_annotations
```

**Parameters:**
- `--img_dir`: Input images directory
- `--ann_dir`: Input annotations directory
- `--out_img_dir`: Output deskewed images directory
- `--out_ann_dir`: Output transformed annotations directory

**What it does:**
1. Detects document skew angle using Hough transform
2. Rotates image to correct orientation
3. Transforms bounding box coordinates accordingly
4. Saves deskewed images and updated annotations

**Expected output:**
```
Processing: 100%|████████████| 1000/1000 [05:23<00:00, 3.09it/s]
✅ Deskewed 1000 images
✅ Saved to: ../data/deskewed_images
```

**Note:** If you deskew your data, update the `src_dir` in `convert_to_yolo.py` to use deskewed images.

---

## Dataset Conversion

### Step 4: Convert to YOLO Format

Convert JSON annotations to YOLO format (normalized coordinates).

**If using original data:**
```bash
python convert_to_yolo.py
```

**If using deskewed data:**
Edit `convert_to_yolo.py` first:
```python
if __name__ == "__main__":
    convert_dataset(
        src_dir="../data/deskewed_images",  # Change this
        out_dir="../data/stage1_dataset"
    )
```

Then run:
```bash
python convert_to_yolo.py
```

**What it does:**
1. Reads images and JSON annotations
2. Converts bbox format: `[x, y, w, h]` → `[x_center, y_center, w, h]` (normalized)
3. Splits data into train/val (90%/10% by default)
4. Creates YOLO directory structure

**Output structure:**
```
data/stage1_dataset/
├── images/
│   ├── train/
│   │   ├── doc_00001.png
│   │   └── ...
│   └── val/
│       ├── doc_00050.png
│       └── ...
└── labels/
    ├── train/
    │   ├── doc_00001.txt
    │   └── ...
    └── val/
        ├── doc_00050.txt
        └── ...
```

**YOLO label format (each line):**
```
class_id x_center y_center width height
```
All values are normalized (0-1 range).

---

## Training Configuration

### Step 5: Configure Dataset YAML

The `dataset.yaml` file defines your dataset configuration:

```yaml
path: stage1_dataset

train: images/train
val: images/val

nc: 5
names: ['Text','Title','List','Table','Figure']
```

**Parameters:**
- `path`: Relative path to dataset root
- `train`: Path to training images (relative to `path`)
- `val`: Path to validation images
- `nc`: Number of classes
- `names`: Class names (must match category order)

**Important:** Ensure the `path` is correct relative to where you run the training command.

---

## Model Training

### Step 6: Initial Training

Start training from a pretrained YOLOv8 model:

**Basic training command:**
```bash
yolo detect train model=yolov8m.pt data=dataset.yaml imgsz=1024 epochs=50 batch=8 lr0=0.001 optimizer=AdamW
```

**Parameter explanation:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `yolov8m.pt` | Pretrained YOLOv8 medium model (downloads automatically) |
| `data` | `dataset.yaml` | Dataset configuration file |
| `imgsz` | `1024` | Input image size (higher = better accuracy, more VRAM) |
| `epochs` | `50` | Number of training epochs |
| `batch` | `8` | Batch size (reduce if CUDA out of memory) |
| `lr0` | `0.001` | Initial learning rate |
| `optimizer` | `AdamW` | Optimizer (AdamW recommended for documents) |

**Model variants:**
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended balance)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large (best accuracy, slowest)

### Step 7: Advanced Training

For better results, use advanced parameters:

```bash
yolo detect train model=yolov8m.pt data=dataset.yaml imgsz=960 epochs=40 batch=6 workers=2 lr0=0.001 optimizer=AdamW device=0 cache=True patience=10 save_period=5
```

**Additional parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `workers` | `2` | Number of data loading workers |
| `device` | `0` | GPU device ID (0 for first GPU, cpu for CPU) |
| `cache` | `True` | Cache images in RAM for faster training |
| `patience` | `10` | Early stopping patience (epochs without improvement) |
| `save_period` | `5` | Save checkpoint every N epochs |
| `project` | `runs/detect` | Project directory |
| `name` | `train` | Experiment name |

### Step 8: Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
yolo detect train resume model=runs/detect/train/weights/last.pt
```

**Note:** This automatically loads all previous training settings.

---

## Training Monitoring

### Step 9: Monitor Training Progress

**Real-time monitoring:**
Training logs are displayed in the terminal:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      3.45G      1.234      0.876      1.123         45        960
  2/50      3.45G      1.156      0.823      1.089         45        960
```

**TensorBoard (optional):**
```bash
tensorboard --logdir runs/detect/train
```

**Training outputs:**
```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
└── args.yaml            # Training arguments
```

### Step 10: Analyze Training Curves

Check `results.png` for:
- **Loss curves:** Should decrease and stabilize
- **mAP curves:** Should increase and plateau
- **Precision/Recall:** Should improve over epochs

**Warning signs:**
- Loss not decreasing → Learning rate too high/low
- Validation loss increasing → Overfitting (reduce epochs or add augmentation)
- Metrics plateauing early → Model capacity too small or data quality issues

---

## Model Evaluation

### Step 11: Validate Model Performance

**Run validation:**
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

**Key metrics:**
- **mAP50:** Mean Average Precision at IoU=0.5
- **mAP50-95:** mAP averaged over IoU thresholds 0.5-0.95
- **Precision:** Percentage of correct predictions
- **Recall:** Percentage of ground truth objects detected

**Target metrics for good performance:**
- mAP50 > 0.85
- mAP50-95 > 0.60
- Precision > 0.80
- Recall > 0.75

### Step 12: Test on Sample Images

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=../data/english_Dataset/doc_00001.png imgsz=960 conf=0.25 save=True
```

**Parameters:**
- `source`: Image, folder, or video path
- `conf`: Confidence threshold (0.25 default)
- `save`: Save annotated images

**Output:** `runs/detect/predict/`

---

## Inference and Export

### Step 13: Run Inference with Deskewing

Use the provided script for production inference:

```bash
python deskew_predict_export.py
```

**Configure paths in the script:**
```python
MODEL_PATH = r"runs/detect/train/weights/best.pt"
SOURCE_DIR = r"path/to/test/images"
OUT_JSON_DIR = r"output/jsons"
OUT_IMG_DIR = r"output/annotated_images"
```

**What it does:**
1. Automatically deskews input images
2. Runs YOLO detection
3. Exports results in JSON format
4. Saves annotated images for verification

**Output JSON format:**
```json
{
  "file_name": "doc_00001.png",
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1
    }
  ]
}
```

### Step 14: Export Model for Deployment

**Export to ONNX (for production):**
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

**Other formats:**
- `format=torchscript` - TorchScript
- `format=engine` - TensorRT (fastest on NVIDIA GPUs)
- `format=coreml` - CoreML (for iOS)

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size: `batch=4` or `batch=2`
- Reduce image size: `imgsz=640` or `imgsz=800`
- Use smaller model: `yolov8s.pt` instead of `yolov8m.pt`
- Disable cache: Remove `cache=True`

**2. Low mAP / Poor Performance**
```
mAP50 < 0.5
```
**Solutions:**
- Check annotation quality (use `get_annoted_image.py`)
- Increase training epochs: `epochs=100`
- Use larger model: `yolov8l.pt`
- Add more training data
- Verify category distribution is balanced
- Enable deskewing preprocessing

**3. Training Not Starting**
```
FileNotFoundError: dataset.yaml not found
```
**Solutions:**
- Ensure `dataset.yaml` is in the current directory
- Check `path` in `dataset.yaml` is correct
- Verify dataset structure matches expected format

**4. Slow Training**
```
Training taking too long
```
**Solutions:**
- Enable caching: `cache=True`
- Reduce workers if CPU bottleneck: `workers=2`
- Use smaller image size: `imgsz=640`
- Check GPU utilization: `nvidia-smi`

**5. Overfitting**
```
Validation loss increasing while training loss decreases
```
**Solutions:**
- Reduce epochs
- Add data augmentation (YOLO does this automatically)
- Increase validation split: `val_split=0.2` in `convert_to_yolo.py`
- Add more diverse training data

---

## Best Practices

### Data Quality
- Minimum 1000+ images for good performance
- Balanced category distribution (within 2x ratio)
- High-quality annotations (no missing or incorrect boxes)
- Diverse document types and layouts

### Training Strategy
- Start with pretrained model (transfer learning)
- Use medium model (`yolov8m.pt`) as baseline
- Train for at least 50 epochs
- Monitor validation metrics, not just training loss
- Save checkpoints regularly

### Hyperparameter Tuning
- Start with default parameters
- Adjust batch size based on GPU memory
- Tune learning rate if loss doesn't decrease
- Use early stopping to prevent overfitting

### Validation
- Always validate on unseen data
- Test on diverse document types
- Check per-class performance (some classes may be weak)
- Visual inspection of predictions

---

## Next Steps

After successful training:

1. **Fine-tune on multilingual data** → See `../finetuning_(multilang)/FINETUNING_STEPS.md`
2. **Deploy as API** → See `../API/README.md`
3. **Containerize with Docker** → See `../Docker_image_folder/DOCKER_IMAGE_STEPS.md`
4. **Benchmark performance** → Use `../utils/resource_calculation.py`

---

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- [Dataset Format Guide](https://docs.ultralytics.com/datasets/detect/)

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `runs/detect/train/`
3. Validate dataset with utility scripts
4. Consult YOLOv8 documentation

---

**Last Updated:** November 2025
**Author:** Multilingual Document Understanding Team
