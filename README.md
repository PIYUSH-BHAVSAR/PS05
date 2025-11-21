
# Multilingual Document Understanding

An advanced Intelligent Multilingual Document Understanding system that leverages computer vision and deep learning to extract structured information from diverse document formats. The system employs YOLOv8-based object detection to identify and localize document elements (text, titles, lists, tables, figures) and provides specialized processors for content extraction.

## Key Features

- Support for multiple document formats (DOC, DOCX, PDF, PPT, JPEG, PNG)
- Robust handling of mixed scripts (English-Arabic, Hindi-English, Chinese-English)
- Automatic skew correction for misaligned documents
- Preservation of document structure, hierarchy, and semantic relationships
- JSON-formatted output with bounding box coordinates and language identification
- FastAPI-based deployment for easy integration
- Docker support for containerized deployment

## Project Structure

```
.
├── API/                              # FastAPI backend for model serving
│   ├── app.py                        # Main API application
│   ├── Dockerfile                    # Docker configuration for API
│   ├── requirements.txt              # API dependencies
│   ├── README.md                     # API documentation
│   └── model/                        # Trained model files
│
├── data/                             # Dataset directory
│   ├── english_Dataset/              # English document dataset
│   └── multilang_Dataset/            # Multilingual document dataset
│
├── training_(english)/               # English model training
│   ├── convert_to_yolo.py            # Convert annotations to YOLO format
│   ├── deskew_and_transform.py       # Image preprocessing and deskewing
│   ├── deskew_predict_export.py      # Prediction with deskewing
│   ├── dataset.yaml                  # YOLO dataset configuration
│   ├── requirements.txt              # Training dependencies
│   └── TRAINING_STEPS.md             # Training documentation
│
├── finetuning_(multilang)/           # Multilingual model fine-tuning
│   ├── convert_to_yolo.py            # Convert annotations to YOLO format
│   ├── deskew_and_transform.py       # Image preprocessing and deskewing
│   ├── multilang.yaml                # Multilingual dataset configuration
│   ├── requirements.txt              # Fine-tuning dependencies
│   └── FINETUNING_STEPS.md           # Fine-tuning documentation
│
├── Docker_image_folder/              # Docker deployment resources
│   ├── Dockerfile                    # Docker configuration
│   ├── best.pt                       # Trained model weights
│   ├── deskew_predict_export.py      # Prediction script
│   ├── DOCKER_IMAGE_STEPS.md         # Docker deployment guide
│   └── datass/                       # Sample data for testing
│
├── frontend/                         # Web interface
│   ├── index.html                    # Main HTML page
│   ├── script.js                     # Frontend JavaScript
│   └── styles.css                    # Styling
│
├── utils/                            # Utility scripts
│   ├── findcatzero.py                # Find categories with zero instances
│   ├── get_annoted_image.py          # Generate annotated images
│   └── resource_calculation.py       # Calculate resource requirements
│
├── runs/                             # Training run outputs
│   └── detect/                       # Detection training results
│
├── deskew_predict_export.py          # Main prediction script with deskewing
├── requirements.txt                  # Root project dependencies
├── README.md                         # This file
└── QNA.md                            # Detailed project Q&A documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd multilingual_doc_understanding
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support with CUDA 11.8:
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio --upgrade
   ```

### Running the API

1. **Navigate to the API directory:**
   ```bash
   cd API
   ```

2. **Install API dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your trained model:**
   - Put your `best.pt` model file in `API/model/` directory

4. **Start the API server:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the API:**
   - API will be available at `http://localhost:8000`
   - API documentation at `http://localhost:8000/docs`

### Using the Frontend

1. Open `frontend/index.html` in a web browser
2. Upload a document image
3. View the detected elements and annotations

## Training Your Own Model

### For English Documents

Navigate to the `training_(english)` directory and follow the steps:

1. **Prepare your dataset:**
   - Place images and JSON annotations in `data/english_Dataset/`

2. **Deskew images (recommended):**
   ```bash
   cd training_(english)
   python deskew_and_transform.py --img_dir ../data/english_Dataset/train --ann_dir ../data/english_Dataset/train --out_img_dir ../data/deskew_images --out_ann_dir ../data/deskew_annotations
   ```

3. **Convert to YOLO format:**
   ```bash
   python convert_to_yolo.py
   ```

4. **Train the model:**
   ```bash
   yolo detect train model=yolov8m.pt data=dataset.yaml imgsz=1024 epochs=50 batch=8 lr0=0.001 optimizer=AdamW
   ```

See `training_(english)/TRAINING_STEPS.md` for detailed instructions.

### For Multilingual Documents

Navigate to the `finetuning_(multilang)` directory:

1. **Prepare multilingual dataset:**
   - Place images and annotations in `data/multilang_Dataset/`

2. **Preprocess and convert:**
   ```bash
   cd finetuning_(multilang)
   python deskew_and_transform.py
   python convert_to_yolo.py
   ```

3. **Fine-tune the model:**
   ```bash
   yolo detect train model=yolov8m.pt data=multilang.yaml imgsz=960 epochs=40 batch=6 workers=2 lr0=0.001 optimizer=AdamW
   ```

See `finetuning_(multilang)/FINETUNING_STEPS.md` for detailed instructions.

## Docker Deployment

### Building the Docker Image

1. **Navigate to Docker folder:**
   ```bash
   cd Docker_image_folder
   ```

2. **Build the image:**
   ```bash
   docker build -t multilingual-doc-understanding .
   ```

3. **Run the container:**
   ```bash
   docker run -p 8000:8000 multilingual-doc-understanding
   ```

See `Docker_image_folder/DOCKER_IMAGE_STEPS.md` for detailed deployment instructions.

### Deploying to HuggingFace Spaces

1. Navigate to the `API` directory
2. Select Docker runtime in HuggingFace Spaces
3. Upload the entire `API` folder
4. The API will be automatically deployed

## Model Inference

### Using the Prediction Script

```bash
python deskew_predict_export.py --source <path_to_image> --model <path_to_best.pt>
```

This script will:
- Automatically deskew the input image
- Run YOLO detection
- Export results with bounding boxes and annotations

### API Endpoints

**Health Check:**
```bash
GET http://localhost:8000/
```

**Single Image Prediction:**
```bash
POST http://localhost:8000/predict/single
Content-Type: multipart/form-data
Body: file=<image_file>
```

Response includes:
- Detected element annotations with bounding boxes
- Category IDs for each element
- Base64-encoded annotated image

## Document Element Categories

The model detects and classifies the following document elements:

1. **Text** - Regular body text paragraphs
2. **Title** - Headings and section titles
3. **List** - Bulleted or numbered lists
4. **Table** - Tabular data structures
5. **Figure** - Images, charts, diagrams, and visual elements

## Key Scripts

### Data Processing
- **convert_to_yolo.py** - Converts JSON annotations to YOLO format with train/val splitting
- **deskew_and_transform.py** - Corrects image skew and transforms bounding box coordinates
- **deskew_predict_export.py** - Runs inference with automatic deskewing

### Utilities
- **utils/findcatzero.py** - Identifies categories with zero instances in dataset
- **utils/get_annoted_image.py** - Generates annotated visualization images
- **utils/resource_calculation.py** - Calculates computational resource requirements

## Advanced Features

### Automatic Deskewing
The system includes robust skew detection and correction:
- Uses Hough transform for angle detection
- Applies affine transformation for rotation
- Automatically adjusts bounding box coordinates
- Improves accuracy on scanned/photographed documents

### Multilingual Support
- Handles mixed-script documents (English-Arabic, Hindi-English, Chinese-English)
- Language-aware text extraction
- Script identification for appropriate OCR routing

### Structured Output
JSON format includes:
- Precise bounding box coordinates
- Element type classification
- Confidence scores
- Language identification
- Hierarchical relationships

## Performance Considerations

### Training Resources
- GPU: NVIDIA T4 or better (16GB+ VRAM recommended)
- RAM: 32GB+ for training, 16GB+ for inference
- Storage: 500GB+ for datasets and model checkpoints

### Inference Optimization
- Batch processing for multiple documents
- GPU acceleration for faster processing
- Configurable image size (960-1024px recommended)
- Confidence threshold tuning (default: 0.25)

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size
- Decrease image size (imgsz parameter)
- Use smaller model variant (yolov8n.pt instead of yolov8m.pt)

**Low Detection Accuracy:**
- Ensure proper image preprocessing (deskewing)
- Check confidence threshold settings
- Verify dataset quality and annotations
- Consider fine-tuning on domain-specific data

**API Connection Issues:**
- Verify port 8000 is not in use
- Check CORS settings in app.py
- Ensure model file exists in API/model/

## Documentation

- **QNA.md** - Comprehensive project documentation and technical details
- **training_(english)/TRAINING_STEPS.md** - English model training guide
- **finetuning_(multilang)/FINETUNING_STEPS.md** - Multilingual fine-tuning guide
- **Docker_image_folder/DOCKER_IMAGE_STEPS.md** - Docker deployment guide
- **API/README.md** - API deployment and usage instructions

## Model Status

- **Train 5 (best.pt)** - Initial trained model included in solution
- **Train 9 (best.pt)** - Updated model with improved performance
- Models available in `runs/detect/` after training

## Contributing

When contributing to this project:
1. Follow the existing code structure
2. Update documentation for new features
3. Test with diverse document types and languages
4. Ensure backward compatibility with existing models

## License

[Add your license information here]

## Contact

[Add contact information or links here]