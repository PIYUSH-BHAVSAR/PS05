# Fine-tuning Steps - Multilingual Document Understanding Model

This guide provides step-by-step instructions for fine-tuning a YOLOv8 model on multilingual documents to improve performance on mixed-script documents (English-Arabic, Hindi-English, Chinese-English, etc.).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Data Preprocessing](#data-preprocessing)
5. [Dataset Conversion](#dataset-conversion)
6. [Fine-tuning Configuration](#fine-tuning-configuration)
7. [Model Fine-tuning](#model-fine-tuning)
8. [Evaluation and Comparison](#evaluation-and-comparison)
9. [Inference on Multilingual Documents](#inference-on-multilingual-documents)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Fine-tuning?

Fine-tuning adapts a pre-trained model (trained on English documents) to perform better on multilingual documents. This approach:
- Requires less training data than training from scratch
- Converges faster (fewer epochs needed)
- Preserves knowledge from English training
- Improves accuracy on mixed-script documents

### When to Fine-tune?

Fine-tune when:
- You have a trained English model with good performance (mAP50 > 0.80)
- You have multilingual documents with different scripts
- Documents contain mixed languages (e.g., English headers with Arabic body text)
- You want to improve model robustness across languages

### Fine-tuning vs Training from Scratch

| Aspect | Fine-tuning | Training from Scratch |
|--------|-------------|----------------------|
| Starting point | Pre-trained model | Random weights |
| Data required | 500+ images | 2000+ images |
| Training time | 20-40 epochs | 50-100 epochs |
| Learning rate | Lower (0.0001-0.001) | Higher (0.001-0.01) |
| Use case | Domain adaptation | New task entirely |

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA RTX 3090 or A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 100GB+ SSD

### Software Requirements

1. **Python 3.8+**
2. **CUDA 11.8+**
3. **Required packages:**

```bash
pip install -r requirements.txt
```

### Pre-trained Model

You need a trained English model before fine-tuning:
- Location: `../training_(english)/runs/detect/train/weights/best.pt`
- Minimum performance: mAP50 > 0.80
- If you don't have one, complete the English training first

---

## Dataset Preparation

### Step 1: Organize Multilingual Dataset

Your multilingual dataset should follow the same structure as English data:

```
data/
└── multilang_Dataset/
    ├── doc_ml_00001.png
    ├── doc_ml_00001.json
    ├── doc_ml_00002.png
    ├── doc_ml_00002.json
    └── ...
```

**JSON Annotation Format:**
```json
{
  "file_name": "doc_ml_00001.png",
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1
    }
  ]
}
```

**Category IDs (same as English):**
- 1: Text
- 2: Title
- 3: List
- 4: Table
- 5: Figure

### Step 2: Dataset Composition Guidelines

For effective fine-tuning, your multilingual dataset should include:

**Language Distribution:**
- 30-40% English documents (for continuity)
- 60-70% target language(s) documents
- Include mixed-script documents

**Document Variety:**
- Different layouts and formats
- Various font styles and sizes
- Handwritten and printed text
- Different quality levels (scanned, photographed)

**Minimum Dataset Size:**
- 500+ images for single additional language
- 1000+ images for multiple languages
- 2000+ images for optimal performance

### Step 3: Validate Dataset Quality

Check category distribution:

```bash
python ../utils/findcatzero.py
```

Edit the script to point to your multilingual dataset:
```python
folder = r"../data/multilang_Dataset"
```

**Quality checks:**
- All 5 categories present
- Balanced distribution across categories
- Sufficient samples per language
- No corrupted JSON files

### Step 4: Visual Inspection

Verify annotations on multilingual documents:

```bash
python ../utils/get_annoted_image.py
```

**Check for:**
- Correct bounding boxes on non-Latin scripts
- Proper handling of right-to-left text (Arabic, Hebrew)
- Accurate table detection in mixed-script documents
- No annotation errors on complex layouts

---

## Data Preprocessing

### Step 5: Deskew Multilingual Images

Deskewing is especially important for multilingual documents:

```bash
python deskew_and_transform.py --img_dir ../data/multilang_Dataset --ann_dir ../data/multilang_Dataset --out_img_dir ../data/multilang_deskewed --out_ann_dir ../data/multilang_deskewed_ann
```

**Parameters:**
- `--img_dir`: Input multilingual images
- `--ann_dir`: Input annotations
- `--out_img_dir`: Output deskewed images
- `--out_ann_dir`: Output transformed annotations

**What it does:**
1. Detects skew angle (works with all scripts)
2. Rotates image to correct orientation
3. Transforms bounding boxes accordingly
4. Preserves annotation integrity

**Expected output:**
```
Processing: 100%|████████████| 800/800 [04:12<00:00, 3.17it/s]
✅ Deskewed 800 multilingual images
✅ Saved to: ../data/multilang_deskewed
```

**Note:** Deskewing algorithm works with all scripts (Latin, Arabic, Chinese, Devanagari, etc.)

---

## Dataset Conversion

### Step 6: Convert to YOLO Format

Convert multilingual annotations to YOLO format:

**If using original data:**
```bash
python convert_to_yolo.py
```

**If using deskewed data:**
Edit `convert_to_yolo.py` first:
```python
if __name__ == "__main__":
    convert_dataset(
        src_dir="../data/multilang_deskewed",  # Change this
        out_dir="../data/multilang_yolo"
    )
```

Then run:
```bash
python convert_to_yolo.py
```

**What it does:**
1. Reads multilingual images and JSON annotations
2. Converts to YOLO normalized format
3. Splits into train/val (90%/10%)
4. Creates YOLO directory structure

**Output structure:**
```
data/multilang_yolo/
├── images/
│   ├── train/
│   │   ├── doc_ml_00001.png
│   │   └── ...
│   └── val/
│       ├── doc_ml_00080.png
│       └── ...
└── labels/
    ├── train/
    │   ├── doc_ml_00001.txt
    │   └── ...
    └── val/
        ├── doc_ml_00080.txt
        └── ...
```

---

## Fine-tuning Configuration

### Step 7: Configure Multilingual Dataset YAML

The `multilang.yaml` file defines your multilingual dataset:

```yaml
path: ../data/multilang_yolo

train: images/train
val: images/val

nc: 5
names:
  0: Text
  1: Title
  2: List
  3: Table
  4: Figure
```

**Important notes:**
- `path` is relative to where you run the training command
- Class names and order must match the English model
- `nc` (number of classes) must be 5

---

## Model Fine-tuning

### Step 8: Basic Fine-tuning

Start fine-tuning from your trained English model:

```bash
yolo detect train \
  model="runs/detect/train3/weights/last.pt" \
  data="multilang.yaml" \
  imgsz=1280 \
  epochs=180 \
  batch=2 \
  workers=2 \
  optimizer=AdamW \
  lr0=0.001 \
  lrf=0.01 \
  device=0 \
  cache=True \
  cos_lr=True \
  patience=60 \
  amp=True \
  mosaic=0.5 \
  mixup=0.05 \
  hsv_h=0.010 \
  hsv_s=0.6 \
  hsv_v=0.4 \
  fliplr=0.15 \
  translate=0.04 \
  scale=0.20

```

**Parameter explanation:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `best.pt` | Your trained English model (not pretrained YOLO) |
| `data` | `multilang.yaml` | Multilingual dataset configuration |
| `imgsz` | `960` | Image size (can be different from English training) |
| `epochs` | `30` | Fewer epochs than training from scratch |
| `batch` | `6` | Batch size (adjust based on GPU memory) |
| `lr0` | `0.0005` | Lower learning rate for fine-tuning |
| `optimizer` | `AdamW` | Optimizer (same as English training) |

**Key differences from initial training:**
- **Lower learning rate:** Prevents catastrophic forgetting
- **Fewer epochs:** Model already has good features
- **Starting from best.pt:** Not from pretrained YOLO

### Step 9: Advanced Fine-tuning

For better multilingual performance:

```bash
yolo detect train model=../training_(english)/runs/detect/train/weights/best.pt data=multilang.yaml imgsz=960 epochs=40 batch=6 workers=2 lr0=0.0005 optimizer=AdamW device=0 cache=True patience=15 save_period=5 freeze=10
```

**Additional parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `freeze` | `10` | Freeze first N layers (preserves low-level features) |
| `patience` | `15` | Early stopping patience |
| `save_period` | `5` | Save checkpoint every 5 epochs |
| `cache` | `True` | Cache images for faster training |

**Layer freezing strategy:**
- `freeze=0`: Fine-tune all layers (more flexible, needs more data)
- `freeze=10`: Freeze backbone (faster, less data needed)
- `freeze=20`: Freeze most layers (minimal adaptation)

### Step 10: Progressive Fine-tuning (Advanced)

For large multilingual datasets, use progressive unfreezing:

**Stage 1: Freeze backbone (20 epochs)**
```bash
yolo detect train model=../training_(english)/runs/detect/train/weights/best.pt data=multilang.yaml epochs=20 lr0=0.001 freeze=15
```

**Stage 2: Unfreeze all layers (20 epochs)**
```bash
yolo detect train resume model=runs/detect/train/weights/last.pt epochs=40 lr0=0.0005 freeze=0
```

This approach:
- Adapts detection head first
- Then fine-tunes entire network
- Prevents catastrophic forgetting
- Often achieves best results

---

## Evaluation and Comparison

### Step 11: Validate Fine-tuned Model

Run validation on multilingual test set:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=multilang.yaml
```

**Key metrics to check:**
- **mAP50:** Should be > 0.80 for good performance
- **mAP50-95:** Should improve from English-only model
- **Per-class metrics:** Check if all categories perform well

### Step 12: Compare with English Model

Test both models on multilingual documents:

**English model:**
```bash
yolo detect val model=../training_(english)/runs/detect/train/weights/best.pt data=multilang.yaml
```

**Fine-tuned model:**
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=multilang.yaml
```

**Expected improvements:**
- 5-15% increase in mAP50 on multilingual documents
- Better detection of non-Latin text
- Improved performance on mixed-script documents
- More robust to different layouts

### Step 13: Visual Comparison

Test on sample multilingual images:

**English model:**
```bash
yolo detect predict model=../training_(english)/runs/detect/train/weights/best.pt source=../data/multilang_Dataset/doc_ml_00001.png save=True project=runs/compare name=english_model
```

**Fine-tuned model:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=../data/multilang_Dataset/doc_ml_00001.png save=True project=runs/compare name=finetuned_model
```

Compare outputs in `runs/compare/` directory.

---

## Inference on Multilingual Documents

### Step 14: Batch Inference

Process multiple multilingual documents:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=../data/multilang_Dataset/ imgsz=960 conf=0.25 save=True save_txt=True
```

**Parameters:**
- `source`: Folder with multilingual images
- `conf`: Confidence threshold (0.25 default)
- `save`: Save annotated images
- `save_txt`: Save YOLO format labels

**Output:** `runs/detect/predict/`

### Step 15: Export for Production

Export fine-tuned model for deployment:

**ONNX format (recommended):**
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

**TensorRT (fastest on NVIDIA GPUs):**
```bash
yolo export model=runs/detect/train/weights/best.pt format=engine
```

**Other formats:**
- `format=torchscript` - TorchScript
- `format=coreml` - CoreML (iOS)
- `format=tflite` - TensorFlow Lite (mobile)

---

## Troubleshooting

### Common Issues

**1. Model Forgetting English Performance**
```
English documents: mAP drops from 0.85 to 0.70
```
**Solutions:**
- Lower learning rate: `lr0=0.0003`
- Freeze more layers: `freeze=15`
- Include more English samples in multilingual dataset (30-40%)
- Reduce training epochs: `epochs=20`

**2. Poor Performance on New Scripts**
```
Arabic/Chinese text: mAP < 0.60
```
**Solutions:**
- Add more samples of target script (500+ images)
- Increase training epochs: `epochs=50`
- Unfreeze all layers: `freeze=0`
- Check annotation quality for new scripts
- Use larger image size: `imgsz=1024`

**3. Overfitting on Multilingual Data**
```
Train mAP: 0.90, Val mAP: 0.65
```
**Solutions:**
- Reduce epochs
- Increase validation split: `val_split=0.2`
- Add more diverse multilingual data
- Use early stopping: `patience=10`
- Freeze more layers: `freeze=15`

**4. Slow Convergence**
```
Loss not decreasing after 20 epochs
```
**Solutions:**
- Increase learning rate: `lr0=0.001`
- Unfreeze more layers: `freeze=5`
- Check if starting model is good (mAP > 0.80)
- Verify dataset quality and balance

**5. Mixed Results Across Languages**
```
English: mAP 0.85, Arabic: mAP 0.60
```
**Solutions:**
- Balance dataset (equal samples per language)
- Train longer: `epochs=50`
- Use language-specific augmentation
- Check annotation quality for underperforming language

---

## Best Practices

### Dataset Strategy
- Include 30-40% English documents for continuity
- Balance samples across all target languages
- Ensure diverse document layouts and quality levels
- Minimum 500+ images per language

### Fine-tuning Strategy
- Always start from a good English model (mAP > 0.80)
- Use lower learning rate than initial training
- Start with frozen layers, then unfreeze progressively
- Monitor both English and multilingual performance

### Hyperparameter Guidelines
- Learning rate: 0.0003 - 0.001 (lower than training)
- Epochs: 20-40 (fewer than training from scratch)
- Freeze: 10-15 layers for small datasets, 0 for large datasets
- Batch size: Same as English training

### Validation Strategy
- Test on both English and multilingual documents
- Check per-language performance
- Visual inspection of predictions on mixed-script documents
- Compare with English-only model

---

## Performance Benchmarking

### Step 16: Benchmark Fine-tuned Model

Measure resource usage:

```bash
python ../utils/resource_calculation.py
```

Edit the script to use your fine-tuned model:
```python
MODEL_PATH = r"runs/detect/train/weights/best.pt"
```

**Expected metrics:**
- Similar inference time as English model
- Slightly higher GPU memory (if model size increased)
- Throughput: 30-50 FPS on RTX 3090

---

## Next Steps

After successful fine-tuning:

1. **Deploy as API** → See `../API/README.md`
2. **Containerize with Docker** → See `../Docker_image_folder/DOCKER_IMAGE_STEPS.md`
3. **Test on production data** → Use real-world multilingual documents
4. **Monitor performance** → Track accuracy across different languages

---

## Advanced Topics

### Continual Learning

For adding new languages without forgetting:
1. Fine-tune on new language with frozen layers
2. Mix old and new language samples (50/50)
3. Use lower learning rate (0.0001)
4. Validate on all languages after each epoch

### Multi-stage Fine-tuning

For multiple languages:
1. Fine-tune on Language A (30 epochs)
2. Fine-tune on Language B starting from A (30 epochs)
3. Fine-tune on mixed A+B+English (20 epochs)

### Language-specific Models

For specialized use cases:
- Train separate models per language
- Use ensemble predictions
- Route documents to appropriate model based on language detection

---

## Additional Resources

- [Transfer Learning Guide](https://docs.ultralytics.com/guides/transfer-learning/)
- [Fine-tuning Best Practices](https://docs.ultralytics.com/guides/model-training-tips/)
- [Multilingual OCR Techniques](https://arxiv.org/abs/2103.06450)

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Compare with English model performance
3. Validate multilingual dataset quality
4. Review fine-tuning logs in `runs/detect/train/`

---

**Last Updated:** November 2025
**Author:** Multilingual Document Understanding Team
