import os, json, argparse, tempfile, cv2
from detector import ProductDetector
from classifier import BrandClassifier
from ocr_engine import OCRExtractor
from shelf_space import estimate_share
from visualize import draw_boxes

def run_pipeline(image_path):
    detector = ProductDetector()
    classifier = BrandClassifier()
    ocr = OCRExtractor()

    result = detector.detect(image_path)

    image = cv2.imread(image_path)
    detections = []

    for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
        x1,y1,x2,y2 = map(int, box)
        crop = image[y1:y2, x1:x2]

        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmp.name, crop)

        brand = classifier.classify(tmp.name)

        detections.append({
            'bbox':[x1,y1,x2,y2],
            'brand':brand
        })

    ocr_text = ocr.extract(image_path)
    shelf_share = estimate_share(detections)

    output = {
        'image_name': os.path.basename(image_path),
        'total_products': len(detections),
        'brands': {},
        'ocr_labels': ocr_text,
        'shelf_share': shelf_share
    }

    for d in detections:
        output['brands'][d['brand']] = output['brands'].get(d['brand'],0)+1

    draw_boxes(image_path, detections, 'outputs/annotated.jpg')

    with open('outputs/results.json','w') as f:
        json.dump(output,f,indent=2)

    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run_pipeline(args.image)
