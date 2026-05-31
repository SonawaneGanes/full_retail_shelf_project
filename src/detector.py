from ultralytics import YOLO


class ProductDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, image_path):
        results = self.model(image_path)
        return results[0]