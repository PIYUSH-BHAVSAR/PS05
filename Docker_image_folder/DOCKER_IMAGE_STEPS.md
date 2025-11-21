# Docker Deployment Guide - Multilingual Document Understanding

This guide provides step-by-step instructions for containerizing and deploying the Multilingual Document Understanding system using Docker.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Setup](#docker-setup)
4. [Building the Docker Image](#building-the-docker-image)
5. [Running the Container](#running-the-container)
6. [Usage Examples](#usage-examples)
7. [GPU Support](#gpu-support)
8. [Deployment Options](#deployment-options)
9. [Troubleshooting](#troubleshooting)
10. [Production Considerations](#production-considerations)

---

## Overview

### What's Included

This Docker deployment packages:
- YOLOv8 trained model (`best.pt`)
- Deskewing and prediction script
- All required dependencies
- CUDA support for GPU acceleration

### Use Cases

- **Batch Processing:** Process large volumes of documents
- **Cloud Deployment:** Deploy on AWS, GCP, Azure
- **Reproducible Environment:** Consistent execution across systems
- **Scalable Infrastructure:** Easy horizontal scaling

---

## Prerequisites

### System Requirements

**For CPU-only deployment:**
- Docker 20.10+
- 8GB+ RAM
- 20GB+ disk space

**For GPU deployment:**
- Docker 20.10+
- NVIDIA GPU with 8GB+ VRAM
- NVIDIA Docker runtime (nvidia-docker2)
- CUDA-compatible GPU drivers
- 16GB+ RAM
- 20GB+ disk space

### Software Installation

**1. Install Docker:**

**Windows:**
- Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
- Enable WSL 2 backend
- Install and restart

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**2. Install NVIDIA Docker (for GPU support):**

**Linux:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows:**
- Docker Desktop automatically supports GPU if NVIDIA drivers are installed
- Ensure WSL 2 has GPU support enabled

**3. Verify Installation:**

```bash
# Check Docker
docker --version

# Check NVIDIA Docker (GPU only)
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

---

## Docker Setup

### Project Structure

The Docker deployment uses files from the parent directory:

```
multilingual_doc_understanding/
├── Docker_image_folder/
│   ├── Dockerfile                    # Docker configuration
│   ├── deskew_predict_export.py      # Inference script
│   ├── best.pt                       # Trained model (place here)
│   ├── datass/                       # Sample test data
│   │   └── hindi_11_page_1.jpg
│   └── DOCKER_IMAGE_STEPS.md         # This file
│
├── requirements.txt                  # Dependencies (parent folder)
└── ...
```

### Step 1: Prepare Model File

Copy your trained model to the Docker folder:

**From English training:**
```bash
copy training_(english)\runs\detect\train\weights\best.pt Docker_image_folder\best.pt
```

**From multilingual fine-tuning:**
```bash
copy finetuning_(multilang)\runs\detect\train\weights\best.pt Docker_image_folder\best.pt
```

**Verify model exists:**
```bash
dir Docker_image_folder\best.pt
```

### Step 2: Review Dockerfile

The Dockerfile references the parent folder's `requirements.txt`:

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY deskew_predict_export.py /app/
COPY best.pt /app/best.pt
COPY ../requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "deskew_predict_export.py"]
```

**Key components:**
- **Base image:** NVIDIA CUDA 12.1.1 runtime
- **Dependencies:** Uses parent folder's `requirements.txt`
- **Model:** Includes `best.pt` in the image
- **Entrypoint:** Runs prediction script automatically

---

## Building the Docker Image

### Step 3: Build the Image

Navigate to the Docker folder and build:

```bash
cd Docker_image_folder
docker build -t multilingual-doc-understanding:latest .
```

**Build options:**

**With custom tag:**
```bash
docker build -t myorg/doc-understanding:v1.0 .
```

**With build arguments:**
```bash
docker build --build-arg CUDA_VERSION=12.1.1 -t multilingual-doc-understanding:latest .
```

**Expected output:**
```
[+] Building 245.3s (12/12) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [internal] load metadata for nvidia/cuda:12.1.1-runtime-ubuntu22.04
 => [1/7] FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
 => [2/7] RUN apt-get update && apt-get install -y python3 python3-pip...
 => [3/7] WORKDIR /app
 => [4/7] COPY deskew_predict_export.py /app/
 => [5/7] COPY best.pt /app/best.pt
 => [6/7] COPY ../requirements.txt /app/
 => [7/7] RUN pip install --no-cache-dir -r requirements.txt
 => exporting to image
 => => naming to docker.io/library/multilingual-doc-understanding:latest
```

### Step 4: Verify Image

Check the built image:

```bash
docker images multilingual-doc-understanding
```

**Expected output:**
```
REPOSITORY                        TAG       IMAGE ID       CREATED         SIZE
multilingual-doc-understanding    latest    abc123def456   2 minutes ago   8.5GB
```

**Inspect image details:**
```bash
docker inspect multilingual-doc-understanding:latest
```

---

## Running the Container

### Step 5: Basic Usage (CPU)

Run inference on a folder of images:

```bash
docker run --rm -v %cd%\input_images:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

**Parameter explanation:**
- `--rm`: Remove container after execution
- `-v %cd%\input_images:/input`: Mount input folder
- `-v %cd%\output:/output`: Mount output folder
- `--source /input`: Input images directory (inside container)
- `--out_json /output/jsons`: Output JSON directory
- `--out_img /output/images`: Output annotated images directory

### Step 6: GPU-Accelerated Usage

For faster processing with GPU:

```bash
docker run --rm --gpus all -v %cd%\input_images:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

**Additional GPU options:**
- `--gpus all`: Use all available GPUs
- `--gpus '"device=0"'`: Use specific GPU (device 0)
- `--gpus 2`: Use 2 GPUs

### Step 7: Interactive Mode

Run container interactively for debugging:

```bash
docker run -it --rm --gpus all -v %cd%\data:/data multilingual-doc-understanding:latest bash
```

Inside the container:
```bash
python3 deskew_predict_export.py --source /data/images --out_json /data/output/jsons --out_img /data/output/images
```

---

## Usage Examples

### Example 1: Process Sample Data

Test with included sample data:

```bash
cd Docker_image_folder
docker run --rm --gpus all -v %cd%\datass:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

**Output:**
```
✅ Loaded model from: best.pt
🧠 Found 1 images in /input
🚀 Starting deskew + YOLO prediction ...
[1/1] Deskewed hindi_11_page_1.jpg by 1.23°

✅ All images processed successfully!
📂 JSONs saved to: /output/jsons
🖼 Annotated deskewed images saved to: /output/images
```

### Example 2: Batch Processing

Process multiple documents:

```bash
docker run --rm --gpus all -v D:\documents\scanned:/input -v D:\documents\processed:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

### Example 3: Production Pipeline

Create a processing script:

**process_documents.bat:**
```batch
@echo off
set INPUT_DIR=D:\production\input
set OUTPUT_DIR=D:\production\output
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%

docker run --rm --gpus all ^
  -v %INPUT_DIR%:/input ^
  -v %OUTPUT_DIR%\%TIMESTAMP%:/output ^
  multilingual-doc-understanding:latest ^
  --source /input ^
  --out_json /output/jsons ^
  --out_img /output/images

echo Processing complete! Results in %OUTPUT_DIR%\%TIMESTAMP%
```

---

## GPU Support

### Verify GPU Access

Check if container can access GPU:

```bash
docker run --rm --gpus all multilingual-doc-understanding:latest python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output:**
```
CUDA available: True
GPU count: 1
```

### GPU Performance Monitoring

Monitor GPU usage during processing:

**Terminal 1 (run container):**
```bash
docker run --rm --gpus all -v %cd%\input:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

**Terminal 2 (monitor GPU):**
```bash
nvidia-smi -l 1
```

### CPU-Only Deployment

For systems without GPU, remove `--gpus all`:

```bash
docker run --rm -v %cd%\input:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

**Note:** CPU inference is 10-20x slower than GPU.

---

## Deployment Options

### Option 1: Local Deployment

Run on local machine for development/testing:

```bash
docker run --rm --gpus all -v %cd%\data:/data multilingual-doc-understanding:latest --source /data/input --out_json /data/output/jsons --out_img /data/output/images
```

### Option 2: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  doc-understanding:
    image: multilingual-doc-understanding:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./input:/input
      - ./output:/output
    command: --source /input --out_json /output/jsons --out_img /output/images
```

Run with:
```bash
docker-compose up
```

### Option 3: Cloud Deployment

**AWS ECS:**
1. Push image to ECR
2. Create ECS task definition with GPU support
3. Deploy as ECS service

**Google Cloud Run:**
1. Push image to GCR
2. Deploy with GPU-enabled instances
3. Configure auto-scaling

**Azure Container Instances:**
1. Push image to ACR
2. Deploy with GPU SKU
3. Configure resource limits

### Option 4: Kubernetes

Create deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-understanding
spec:
  replicas: 3
  selector:
    matchLabels:
      app: doc-understanding
  template:
    metadata:
      labels:
        app: doc-understanding
    spec:
      containers:
      - name: doc-understanding
        image: multilingual-doc-understanding:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: doc-data-pvc
```

---

## Troubleshooting

### Issue 1: Build Fails - requirements.txt Not Found

**Error:**
```
COPY failed: file not found in build context
```

**Solution:**
Ensure you're building from the Docker_image_folder and the parent requirements.txt exists:
```bash
cd Docker_image_folder
dir ..\requirements.txt
docker build -t multilingual-doc-understanding:latest .
```

### Issue 2: GPU Not Detected

**Error:**
```
CUDA available: False
```

**Solutions:**
1. Verify NVIDIA Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
   ```

2. Check Docker daemon configuration (`/etc/docker/daemon.json`):
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

3. Restart Docker:
   ```bash
   sudo systemctl restart docker
   ```

### Issue 3: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Limit GPU memory:
   ```bash
   docker run --rm --gpus all --memory=8g --memory-swap=8g ...
   ```

2. Process images in smaller batches

3. Reduce image size in script (edit `IMG_SIZE = 960` to `640`)

### Issue 4: Permission Denied on Output

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/output/jsons'
```

**Solutions:**
1. Create output directories first:
   ```bash
   mkdir output\jsons output\images
   ```

2. Run with user permissions:
   ```bash
   docker run --rm --user $(id -u):$(id -g) ...
   ```

### Issue 5: Slow Performance

**Symptoms:**
- Processing takes too long
- Low GPU utilization

**Solutions:**
1. Verify GPU is being used:
   ```bash
   nvidia-smi
   ```

2. Check if running on CPU instead of GPU

3. Increase batch processing (modify script)

4. Use SSD for input/output volumes

---

## Production Considerations

### Performance Optimization

**1. Image Size Reduction:**
- Use multi-stage builds
- Remove unnecessary dependencies
- Clean up apt cache

**2. Caching:**
- Cache model in memory
- Reuse container instances
- Use volume mounts for data

**3. Parallel Processing:**
- Run multiple containers
- Use container orchestration
- Implement queue-based processing

### Security Best Practices

**1. Non-root User:**
Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

**2. Read-only Filesystem:**
```bash
docker run --read-only --tmpfs /tmp ...
```

**3. Resource Limits:**
```bash
docker run --cpus=4 --memory=8g --gpus 1 ...
```

### Monitoring and Logging

**1. Container Logs:**
```bash
docker logs <container_id>
```

**2. Export Logs:**
```bash
docker run --log-driver=json-file --log-opt max-size=10m ...
```

**3. Health Checks:**
Add to Dockerfile:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 -c "import torch; assert torch.cuda.is_available()"
```

### Scaling Strategy

**Horizontal Scaling:**
- Deploy multiple container instances
- Use load balancer for distribution
- Implement queue-based job processing

**Vertical Scaling:**
- Increase GPU memory
- Use larger instance types
- Optimize batch sizes

---

## Advanced Usage

### Custom Configuration

Create a config file and mount it:

**config.json:**
```json
{
  "conf_threshold": 0.25,
  "img_size": 960,
  "save_annotated": true
}
```

**Run with config:**
```bash
docker run --rm -v %cd%\config.json:/app/config.json -v %cd%\input:/input -v %cd%\output:/output multilingual-doc-understanding:latest --source /input --out_json /output/jsons --out_img /output/images
```

### Batch Processing Script

Create automated processing pipeline:

**process_batch.py:**
```python
import os
import subprocess
from pathlib import Path

input_dir = Path("D:/documents/input")
output_dir = Path("D:/documents/output")

for batch_folder in input_dir.iterdir():
    if batch_folder.is_dir():
        batch_output = output_dir / batch_folder.name
        batch_output.mkdir(exist_ok=True)
        
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{batch_folder}:/input",
            "-v", f"{batch_output}:/output",
            "multilingual-doc-understanding:latest",
            "--source", "/input",
            "--out_json", "/output/jsons",
            "--out_img", "/output/images"
        ]
        
        subprocess.run(cmd)
        print(f"Processed {batch_folder.name}")
```

---

## Next Steps

After successful Docker deployment:

1. **API Deployment** → See `../API/README.md` for REST API setup
2. **Cloud Deployment** → Deploy to AWS/GCP/Azure
3. **Monitoring Setup** → Implement logging and metrics
4. **CI/CD Pipeline** → Automate builds and deployments

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Guide](https://github.com/NVIDIA/nvidia-docker)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Container Security](https://docs.docker.com/engine/security/)

---

**Last Updated:** November 2025
**Author:** Multilingual Document Understanding Team
