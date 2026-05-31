import cv2
import easyocr


class OCRExtractor:

    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return []

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        try:
            results = self.reader.readtext(
                gray,
                detail=0
            )

            return results

        except Exception as e:

            print(
                f"OCR Error: {e}"
            )

            return []