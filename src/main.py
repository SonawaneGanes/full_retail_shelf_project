import os
import json
import cv2
import tempfile
import argparse

from detector import ProductDetector
from classifier import BrandClassifier
from visualize import draw_boxes
from ocr import OCRExtractor
from metrics import (
    generate_metrics,
    estimate_shelf_share
)


def run_pipeline(image_path):

    detector = ProductDetector()

    classifier = BrandClassifier()

    ocr = OCRExtractor()

    image = cv2.imread(image_path)

    results = detector.detect(image_path)

    detections = []

    for box in results.boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = map(int, box)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        )

        cv2.imwrite(
            temp_file.name,
            crop
        )

        cls_result = classifier.classify(
            temp_file.name
        )

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "brand": cls_result["brand"],
            "confidence":
                cls_result["confidence"]
        })

    ocr_text = ocr.extract(image_path)

    brand_counts = generate_metrics(
        detections
    )

    shelf_share = estimate_shelf_share(
        detections
    )

    output = {

        "image_name":
            os.path.basename(image_path),

        "total_products":
            len(detections),

        "brands":
            brand_counts,

        "ocr_labels":
            ocr_text,

        "shelf_share":
            shelf_share
    }

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    json_path = (
        f"outputs/results_{image_name}.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    annotated_path = (
        f"outputs/annotated_{image_name}.jpg"
    )

    draw_boxes(
        image_path,
        detections,
        annotated_path
    )

    print(
        f"Saved: {annotated_path}"
    )

    print(
        f"Saved: {json_path}"
    )

    print(
        json.dumps(
            output,
            indent=4
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True
    )

    args = parser.parse_args()

    run_pipeline(
        args.image
    )