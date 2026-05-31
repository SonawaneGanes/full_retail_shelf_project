# Retail Shelf Intelligence

## Overview

This project implements a Retail Shelf Intelligence pipeline that analyzes shelf images and generates actionable insights using Computer Vision and OCR techniques.

The system performs:

* Product/Object Detection
* Brand/Product Classification
* OCR Extraction from Labels and Price Tags
* Shelf Space Estimation
* Business Metrics Generation

---

## Architecture

```text
Input Shelf Image
        │
        ▼
YOLOv8 Product Detection
        │
        ▼
Crop Detected Products
        │
        ▼
OpenAI CLIP Classification
        │
        ▼
EasyOCR Text Extraction
        │
        ▼
Shelf Share Calculation
        │
        ▼
JSON Output + Annotated Image
```

---

## Technologies Used

| Component              | Technology  |
| ---------------------- | ----------- |
| Object Detection       | YOLOv8      |
| Product Classification | OpenAI CLIP |
| OCR                    | EasyOCR     |
| Image Processing       | OpenCV      |
| Deep Learning          | PyTorch     |

---

## Project Structure

```text
retail_shelf_intelligence_project/
│
├── data/
├── outputs/
├── src/
│   ├── detector.py
│   ├── classifier.py
│   ├── ocr.py
│   ├── metrics.py
│   ├── visualize.py
│   ├── brands.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## Installation

## Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
---

## Usage

Run the pipeline on a shelf image:

```bash
python src/main.py --image data/img_1.jpg
python src/main.py --image data/img_2.jpg
python src/main.py --image data/img_3.jpg
```

---

## Outputs

For each image the system generates:

### Annotated Visualization

```text
outputs/annotated_img_1.jpg
```

Contains:

* Bounding Boxes
* Predicted Brand Labels
* Confidence Scores

### JSON Report

```text
outputs/results_img_1.json
```

Example:

```json
{
    "image_name": "img_1.jpg",
    "total_products": 62,
    "brands": {
        "Amul": 11,
        "Tropicana": 16
    },
    "ocr_labels": ["Amul", "Tropicana"],
    "shelf_share": {
        "Amul": 6.24,
        "Tropicana": 73.42
    }
}
```

---

## Model Selection Rationale

### YOLOv8

Selected for fast and accurate real-time object detection with minimal setup requirements.

### OpenAI CLIP

Used for zero-shot product and brand classification without requiring custom training data.

### EasyOCR

Provides robust OCR capabilities for extracting product names, labels, and price information from shelf images.

---

## Assumptions

* Products are visible and reasonably distinguishable in shelf images.
* Shelf share is approximated using detected bounding-box area.
* Brand classification is performed using zero-shot CLIP prompts.

---

## Limitations

* YOLOv8 COCO model is not specifically trained on retail SKU datasets.
* Classification accuracy depends on the provided CLIP label set.
* OCR performance may degrade on blurred or low-resolution images.

---

## Future Improvements

* Fine-tune YOLO on retail shelf datasets.
* Train a retail-specific brand classifier.
* Improve OCR using image enhancement techniques.
* Add SKU-level product recognition.
* Generate dashboard-based analytics.
#
