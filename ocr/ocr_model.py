import easyocr
from PIL.Image import Image


class OcrModel:
    def __init__(self):
        self.image = None

    def get_text(self, image) -> str:
        raise NotImplementedError("Этот метод должен быть переопределен в дочернем классе")


class EasyOcrModel(OcrModel):
    def __init__(self):
        super().__init__()
        self.reader = easyocr.Reader(['ru', 'en'])

    def get_text(self, image):
        self.image = image
        results = self.reader.readtext(image)
        results_list = [text for (bbox, text, prob) in results]

        return ' '.join(results_list)


class TesseractOcrModel(OcrModel):
    def __init__(self):
        super().__init__()

    def get_text(self, image):
        self.image = image
        result = pytesseract.image_to_string(image, lang='rus+eng')
        return result.replace('\n', ' ')


class SuryaOcrModel(OcrModel):
    def __init__(self):
        super().__init__()
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    def get_text(self, image):
        if isinstance(image, Image.Image):
            self.image = image
        elif isinstance(image, np.ndarray):
            self.image = Image.fromarray(image)
        else:
            raise TypeError("Переданный объект должен быть PIL Image или numpy array.")
        langs = ["ru", "en"]
        predictions = self.recognition_predictor([self.image], [langs], self.detection_predictor)
        predict_text = [item.text for item in predictions[0].text_lines]

        return ' '.join(predict_text)
