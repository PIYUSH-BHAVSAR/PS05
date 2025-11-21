# YOLO Deskew FastAPI - API Documentation

A production-ready FastAPI application for multilingual document understanding using YOLOv8. The API automatically deskews document images and detects document elements (Text, Title, List, Table, Figure).

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running Locally](#running-locally)
5. [API Endpoints](#api-endpoints)
6. [Usage Examples](#usage-examples)
7. [Deployment](#deployment)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

---

## Features

- **Automatic Deskewing:** Corrects tilted/rotated documents before processing
- **Multi-element Detection:** Detects Text, Title, List, Table, and Figure elements
- **RESTful API:** Easy integration with any application
- **CORS Enabled:** Supports cross-origin requests for web applications
- **Base64 Image Response:** Returns annotated images for visualization
- **JSON Output:** Structured annotations with bounding boxes and category IDs
- **GPU Acceleration:** Leverages CUDA for fast inference
- **Production Ready:** Includes error handling and logging

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: NVIDIA GPU with 4GB+ VRAM (optional but recommended)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 2060 or better)

### Software Requirements

- **Python 3.8+**
- **CUDA 11.8+** (for GPU acceleration)
- **Trained YOLO model** (`best.pt` file)

---

## Installation

### Step 1: Navigate to API Directory

```bash
cd API
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `pillow` - Image processing
- `opencv-python-headless` - Computer vision operations
- `ultralytics` - YOLOv8 framework
- `numpy` - Numerical operations

### Step 4: Add Your Trained Model

Place your trained YOLO model in the `model` directory:

```bash
mkdir model
# Copy your best.pt file to model/
copy path\to\your\best.pt model\best.pt
```

**Model requirements:**
- Must be a YOLOv8 detection model
- Trained on 5 classes: Text, Title, List, Table, Figure
- Recommended: Model from `training_(english)` or `finetuning_(multilang)`

---

## Running Locally

### Start the API Server

**Development mode (with auto-reload):**
```bash
uvicorn app:app --reload
```

**Production mode:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**With custom host and port:**
```bash
uvicorn app:app --host 127.0.0.1 --port 5000 --reload
```

### Verify API is Running

Open your browser and navigate to:
- **API Root:** http://localhost:8000/
- **Interactive Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

You should see:
```json
{
  "message": "API is running ✅"
}
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /`

**Description:** Check if the API is running

**Response:**
```json
{
  "message": "API is running ✅"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

### 2. Single Image Prediction

**Endpoint:** `POST /predict/single`

**Description:** Process a single document image and return detected elements

**Request:**
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body:** `file` (image file)

**Supported formats:** PNG, JPG, JPEG

**Response:**
```json
{
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1
    },
    {
      "bbox": [x, y, width, height],
      "category_id": 2
    }
  ],
  "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Category IDs:**
- `1` - Text
- `2` - Title
- `3` - List
- `4` - Table
- `5` - Figure

**Bounding Box Format:**
- `[x, y, width, height]` in pixels
- `x, y` - Top-left corner coordinates
- `width, height` - Box dimensions

---

## Usage Examples

### Using cURL

**Basic request:**
```bash
curl -X POST "http://localhost:8000/predict/single" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.png"
```

**Save response to file:**
```bash
curl -X POST "http://localhost:8000/predict/single" \
  -F "file=@document.png" \
  -o response.json
```

### Using Python Requests

```python
import requests
import json
import base64
from PIL import Image
import io

# API endpoint
url = "http://localhost:8000/predict/single"

# Open and send image
with open("document.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Parse response
data = response.json()

# Get annotations
annotations = data["annotations"]
print(f"Found {len(annotations)} elements:")
for ann in annotations:
    bbox = ann["bbox"]
    cat_id = ann["category_id"]
    print(f"  Category {cat_id}: bbox={bbox}")

# Decode and save annotated image
img_base64 = data["annotated_image_base64"]
img_bytes = base64.b64decode(img_base64)
img = Image.open(io.BytesIO(img_bytes))
img.save("annotated_output.png")
print("Annotated image saved!")
```

### Using JavaScript (Fetch API)

```javascript
async function predictDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/predict/single', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  
  // Display annotations
  console.log('Annotations:', data.annotations);
  
  // Display annotated image
  const imgElement = document.getElementById('result');
  imgElement.src = `data:image/png;base64,${data.annotated_image_base64}`;
}

// Usage with file input
document.getElementById('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  predictDocument(file);
});
```

### Using Postman

1. **Create new request:** POST
2. **URL:** `http://localhost:8000/predict/single`
3. **Body tab:** Select "form-data"
4. **Add key:** `file` (change type to "File")
5. **Select file:** Choose your document image
6. **Send request**

---

## Deployment

### Deploy on HuggingFace Spaces

HuggingFace Spaces provides free hosting for ML applications.

**Steps:**

1. **Create a new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Docker" as the SDK
   - Choose GPU (optional but recommended)

2. **Prepare files:**
   - Ensure `Dockerfile` exists in API folder
   - Ensure `model/best.pt` is present
   - All dependencies in `requirements.txt`

3. **Upload files:**
   - Upload the entire `API` folder to your Space
   - Or connect via Git and push files

4. **Configure Space:**
   - Set hardware: CPU Basic (free) or GPU (paid)
   - Space will automatically build and deploy

5. **Access your API:**
   - URL: `https://your-username-space-name.hf.space`
   - Docs: `https://your-username-space-name.hf.space/docs`

**Note:** For HuggingFace deployment, create a `Dockerfile` in the API folder:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY app.py .
COPY model/ model/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Deploy on AWS/GCP/Azure

**Using Docker (recommended):**

1. Build Docker image (see Docker deployment guide)
2. Push to container registry
3. Deploy to cloud service (ECS, Cloud Run, App Service)

**Using VM:**

1. Set up VM with Python and CUDA
2. Clone repository
3. Install dependencies
4. Run with systemd or supervisor

---

## Configuration

### Modify Model Path

Edit `app.py`:

```python
MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
```

### Adjust Inference Parameters

Edit `app.py` in the `process_image` function:

```python
results = model.predict(
    deskewed_cv,
    imgsz=960,      # Image size (640, 960, 1024)
    conf=0.25,      # Confidence threshold (0.1 - 0.9)
    save=False,
    verbose=False
)
```

**Parameters:**
- `imgsz`: Higher = better accuracy, slower inference
- `conf`: Lower = more detections, more false positives

### Enable/Disable CORS

Edit `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Change to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Security note:** In production, replace `["*"]` with specific allowed origins.

### Add Authentication (Optional)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

@app.post("/predict/single")
async def predict_single(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    # ... existing code
```

---

## Troubleshooting

### Common Issues

**1. Model not found error**
```
FileNotFoundError: model/best.pt not found
```
**Solution:**
- Ensure `model/best.pt` exists
- Check MODEL_PATH in `app.py`
- Verify file permissions

**2. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce image size: `imgsz=640`
- Process images sequentially (already done)
- Use CPU mode: Set `device='cpu'` in model.predict()

**3. Port already in use**
```
ERROR: [Errno 98] Address already in use
```
**Solution:**
- Use different port: `uvicorn app:app --port 8001`
- Kill existing process: `lsof -ti:8000 | xargs kill -9` (Linux/Mac)
- Or: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F` (Windows)

**4. Slow inference**
```
Taking >5 seconds per image
```
**Solution:**
- Enable GPU: Ensure CUDA is installed
- Reduce image size: `imgsz=640`
- Use smaller model variant
- Check GPU utilization: `nvidia-smi`

**5. Import errors**
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:**
- Activate virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (must be 3.8+)

---

## Performance Optimization

### GPU Acceleration

Ensure CUDA is properly configured:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### Batch Processing (Future Enhancement)

For processing multiple images:

```python
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        # Process each file
        annotations, annotated_img = process_image(image)
        results.append({"filename": file.filename, "annotations": annotations})
    return results
```

### Caching

For repeated requests on same images:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_predict(image_hash):
    # Prediction logic
    pass
```

### Load Balancing

For high traffic, use multiple workers:

```bash
uvicorn app:app --workers 4 --host 0.0.0.0 --port 8000
```

**Note:** Each worker loads the model, so ensure sufficient GPU memory.

---

## API Response Times

**Expected performance (RTX 3090):**
- Small images (< 1MB): 50-100ms
- Medium images (1-3MB): 100-200ms
- Large images (> 3MB): 200-500ms

**Breakdown:**
- Deskewing: 20-50ms
- YOLO inference: 30-100ms
- Post-processing: 10-20ms

---

## Monitoring and Logging

### Add Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)):
    logger.info(f"Processing file: {file.filename}")
    # ... existing code
    logger.info(f"Found {len(annotations)} annotations")
    return response
```

### Health Check Endpoint

Already included at `GET /`

### Metrics Endpoint (Optional)

```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Security Best Practices

1. **Validate file types:**
   - Check file extensions
   - Verify file content (magic bytes)

2. **Limit file size:**
   ```python
   MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
   ```

3. **Rate limiting:**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict/single")
   @limiter.limit("10/minute")
   async def predict_single(...):
       pass
   ```

4. **Use HTTPS in production**

5. **Implement authentication** (see Configuration section)

---

## Testing

### Test with Sample Image

```bash
# Download sample document
curl -o test_doc.png https://example.com/sample_document.png

# Test API
curl -X POST "http://localhost:8000/predict/single" \
  -F "file=@test_doc.png" \
  -o result.json

# Check result
cat result.json
```

### Automated Testing

```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API is running ✅"

def test_predict_single():
    with open("test_image.png", "rb") as f:
        response = client.post(
            "/predict/single",
            files={"file": ("test.png", f, "image/png")}
        )
    assert response.status_code == 200
    assert "annotations" in response.json()
```

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Docker Deployment Guide](../Docker_image_folder/DOCKER_IMAGE_STEPS.md)

---

## Support

For issues or questions:
1. Check troubleshooting section
2. Review API logs
3. Test with sample images
4. Verify model and dependencies

---

**Last Updated:** November 2025
**Version:** 1.0.0
**Author:** Multilingual Document Understanding Team
