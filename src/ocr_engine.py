import easyocr

class OCRExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract(self, image_path):
        return self.reader.readtext(image_path, detail=0)
