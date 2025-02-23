import cv2
from PIL import Image
import numpy as np


class ImgPreprocessing:
    def __init__(self, path=None):
        """
        Инициализация класса с путём к изображению.
        Обработка в формате np.array

        :param path: путь к исходному изображению.
        """
        if path is not None:
            self.image = self.load_image(path)  # Загрузка изображения с учетом формата
        else:
            self.image = None
        self.original_image = None
        self.image_rgb = None

    def load_image(self, path):
        """
        Проверяет формат изображения и преобразует в формат .jpg, если необходимо.
        """
        # Проверка формата файла
        if path.lower().endswith(('.png', '.bmp', '.gif', '.tiff', '.jpeg')):
            # Если формат не jpg, преобразуем в jpg
            image = cv2.imread(path)
            jpg_path = path.rsplit('.', 1)[0] + '.jpg'  # Изменяем расширение на .jpg
            cv2.imwrite(jpg_path, image)  # Сохраняем как jpg
            return cv2.imread(jpg_path)  # Загружаем сохранённое изображение
        else:
            return cv2.imread(path)  # Если уже .jpg, просто загружаем изображение

    def image_fromarray(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            pass

        self.image = image.copy()

    def convert_to_rgb(self):
        """
        Преобразует изображение в формат RGB.
        """
        if self.image.shape[2] == 4:  # Если изображение имеет альфа-канал
            self.image_rgb = Image.fromarray(self.image).convert('RGB')
            self.image_rgb = np.array(self.image_rgb)
        else:
            self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Преобразуем из BGR в RGB
        return self.image_rgb

    def convert_to_grey(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def thresholding(self):
        """
        Применяет пороговую обработку (градации серого и метод Оцу).
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Преобразование в градации серого
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = thresholded_image
        return self.image

    def preprocess_morphological_transform(self):
        """
        Удаляет артефакты и шумы с помощью морфологических преобразований.
        """
        kernel = np.ones((2, 2), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        return self.image

    def calculate_bisector_angle(self):
        #gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
        # Сегментация строк
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        angles = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            # PCA для определения ориентации
            data = cnt[:, 0, :].astype(np.float32)
            mean, eigenvectors = cv2.PCACompute(data, mean=None)
            angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
            # Корректировка угла
            if angle > 45:
                angle -= 90
            elif angle < -45:
                angle += 90
            angles.append(angle)
        if not angles:
            return 0.0
        # Векторное усреднение углов
        vectors = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in angles]
        avg_vector = np.mean(vectors, axis=0)
        bisector_angle = np.arctan2(avg_vector[1], avg_vector[0]) * 180 / np.pi
        return bisector_angle

    def rotate_image(self, angle=None):
        """
        Поворачивает изображение на заданный угол.

        :param angle: угол поворота в градусах.
        :return: повернутое изображение.
        если угол не передан - вычисляется автоматом
        """
        if angle is None:
            angle = self.calculate_bisector_angle()
            print(angle)
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w - w) // 2
        M[1, 2] += (new_h - h) // 2
        self.image = cv2.warpAffine(self.image, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        return self.image

    def change_contrast(self, factor=1.2):
        """
        Изменяет контрастность изображения.

        :param factor: коэффициент контрастности.
        :return: изображение с измененной контрастностью.
        """
        self.image = cv2.convertScaleAbs(self.image, alpha=factor, beta=0)
        print(f'Применена регулировка контрастности {factor}')
        return self.image

    def increase_resolution(self, scale_factor=1.0):
        """
        Увеличивает разрешение изображения с помощью интерполяции.

        :param scale_factor: коэффициент увеличения.
        :return: изображение с увеличенным разрешением.
        """
        width = int(self.image.shape[1] * scale_factor)
        height = int(self.image.shape[0] * scale_factor)
        self.image = cv2.resize(self.image, (width, height), interpolation=cv2.INTER_CUBIC)
        print(f'Применено масштабирование {scale_factor}')
        return self.image

    def adjust_brightness(self, factor=1.0):
        """
        Регулирует яркость изображения.

        :param factor: коэффициент яркости (значение > 1 увеличивает яркость, < 1 уменьшает).
        :return: изображение с отрегулированной яркостью.
        """
        self.image = cv2.convertScaleAbs(self.image, alpha=1, beta=factor * 100)
        print(f'Применена регулировка яркости {factor}')
        return self.image

    def run(self, thresholding=False, brightness=True, rotate=True, upscale=True, contrast=True,
            preprocess_morphological_transform=False, brightness_factor=1.0, contrast_factor=1.0,
            rotation_angle=None, scale_factor=1.0):
        """
        Метод для запуска всех операций предобработки с возможностью их включения/отключения.

        Параметры:
            - thresholding: True/False, для включения/отключения пороговой обработки.
            - brightness: True/False, для включения/отключения изменения яркости.
            - rotate: True/False, для включения/отключения вращения изображения.
            - upscale: True/False, для включения/отключения увеличения разрешения.
            - contrast: True/False, для включения/отключения изменения контрастности.
            - preprocess_morphological_transform: True/False, для включения/отключения морфологических преобразований.
            - brightness_factor: коэффициент изменения яркости (по умолчанию 1.0).
            - contrast_factor: коэффициент изменения контрастности (по умолчанию 1.0).
            - rotation_angle: угол поворота изображения (по умолчанию 0).
            - scale_factor: коэффициент увеличения разрешения (по умолчанию 1.0).

        Возвращает:
            - Обработанное изображение.
        """
        self.original_image = self.image.copy()
        proc_image = self.image

        if thresholding:
            proc_image = self.thresholding()

        if brightness:
            proc_image = self.adjust_brightness(brightness_factor)

        if rotate:
            proc_image = self.rotate_image(rotation_angle)

        if upscale:
            proc_image = self.increase_resolution(scale_factor)

        if contrast:
            proc_image = self.change_contrast(contrast_factor)

        if preprocess_morphological_transform:
            proc_image = self.preprocess_morphological_transform()

        return proc_image


if __name__ == '__main__':
    image_processor = ImgPreprocessing()
    image_cv = cv2.imread('test.JPG')
    image_processor.image_fromarray(image_cv)
    # обработка изображения с учетом параметров
    processed_image = image_processor.run(
        rotate=True, rotation_angle=None,
        brightness_factor=2,
        contrast_factor=0.1,
        scale_factor=.1
    )
    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_pil.show()
