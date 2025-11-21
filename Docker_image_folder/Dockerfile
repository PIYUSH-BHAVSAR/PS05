FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Make work directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy project files
# ----------------------------
COPY deskew_predict_export.py /app/
COPY best.pt /app/best.pt
COPY requirements.txt /app/

# ----------------------------
# Install PyTorch (CUDA 12 runtime)
# ----------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ----------------------------
# Install YOLO + OpenCV + Numpy
# ----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Run script
# ----------------------------
ENTRYPOINT ["python3", "deskew_predict_export.py"]
