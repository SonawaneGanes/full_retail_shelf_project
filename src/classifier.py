from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

from brands import BRAND_LABELS


class BrandClassifier:

    def __init__(self):

        self.labels = BRAND_LABELS

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def classify(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)

        pred_idx = probs.argmax().item()

        return {
            "brand": self.labels[pred_idx],
            "confidence": round(
                probs[0][pred_idx].item(),
                3
            )
        }