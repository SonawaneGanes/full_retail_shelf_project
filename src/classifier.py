from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class BrandClassifier:
    def __init__(self):
        self.labels = ['Coca-Cola','Pepsi','Lays','Doritos','Amul','Tropicana','Other']
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def classify(self, crop_path):
        image = Image.open(crop_path)
        inputs = self.processor(text=self.labels, images=image, return_tensors='pt', padding=True)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
        return self.labels[int(torch.argmax(probs))]
