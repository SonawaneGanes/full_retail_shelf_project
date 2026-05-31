import cv2

def draw_boxes(image_path, detections, output_path):

    image = cv2.imread(image_path)

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]

        brand = det["brand"]

        confidence = det["confidence"]

        label = f"{brand} ({confidence:.2f})"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, image)
    