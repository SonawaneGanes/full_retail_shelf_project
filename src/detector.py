from ultralytics import YOLO

class ProductDetector:
    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model)

    def detect(self, image_path):
        return self.model(image_path)[0]
