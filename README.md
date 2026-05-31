# Retail Shelf Intelligence Pipeline

## Objective
Analyze retail shelf images and generate:
- Product detection
- Brand classification
- OCR extraction
- Shelf-space estimation
- Business metrics

## Architecture

Shelf Image
    |
    v
YOLOv8 Detection
    |
    v
Crop Products
    |
    +--> CLIP Brand Classification
    |
    +--> EasyOCR Shelf Labels
    |
    v
Shelf Space Estimation
    |
    v
Business Metrics JSON
    |
    v
Annotated Visualization

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py --image data/img_1.jpg
python src/main.py --image data/img_2.jpg
python src/main.py --image data/img_3.jpg
```

## Output
- Annotated image
- JSON metrics
