import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageKit:
    """
    Класс для работы с изображениями в форматах PIL и Open CV
    Поле класса хранит изображение в формате cv
    Показывает изображение в формате PIL либо matplotlib
    """

    def __init__(self):
        self._image_cv = None

    def load_from_file(self, image_path, to_pil=False):
        try:
            self._image_cv = cv2.imread(image_path)  # Загружаем изображение в формате OpenCV
            return self.get_image(to_pil=to_pil)
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")

    def save_to_file(self, image_path, image=None):
        if image is not None:
            self.set_image(image)
        if self._image_cv is not None:
            try:
                image = self._image_cv
                cv2.imwrite(image_path, image)
            except Exception as e:
                print(f"Ошибка при сохранении изображения: {e}")
        else:
            print('Нет изображения для сохранения')

    def to_pil(self, image):
        if image is None:
            if self._image_cv is not None:
                image = self._image_cv
            else:
                print('Нет изображения для показа')
                return
        if isinstance(image, np.ndarray):  # Проверяем, является ли изображение cv
            # Преобразуем BGR (OpenCV) в RGB (PIL)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)

        return image

    def show_image_pil(self, image=None):
        image_pil = self.to_pil(image)      # преобразуем в pil если требуется
        image_pil.show()

    def show_image_plt(self, image=None):
        image_pil = self.to_pil(image)
        plt.imshow(image_pil)
        plt.axis('off')
        plt.show()

    def get_image(self, to_pil=False):
        if to_pil:
            # Преобразуем BGR (OpenCV) в RGB (PIL)
            image_rgb = cv2.cvtColor(self._image_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)

        return self._image_cv

    def set_image(self, image):
        if isinstance(image, Image.Image):  # Проверяем, является ли изображение PIL
            # Конвертируем PIL в OpenCV
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Сохраняем изображение OpenCV в поле
        self._image_cv = image


if __name__ == '__main__':
    # Создаем объект ImageKit
    image_kit = ImageKit()

    # Загружаем изображение
    file_path1 = 'test.jpg'  # Замените на путь к вашему файлу
    file_path2 = 'test2.jpg'  # Замените на путь к вашему файлу
    print("Загрузка изображения...")
    # Получаем изображение в формате PIL
    pil_image = image_kit.load_from_file(file_path1, to_pil=True)
    # Загружаем второе изображение
    image_kit.load_from_file(file_path2)

    # Отображаем изображение test2.jpg в формате plt
    print("Отображение изображения  в формате plt...")
    image_kit.show_image_plt()

    # Отображаем изображение в формате PIL
    print("Отображение изображения в формате PIL...")
    image_kit.show_image_pil(pil_image)

    # Сохраняем изображение test2.jpg
    output_path_cv = 'output_cv.jpg'
    print(f"Сохранение изображения в формате OpenCV в {output_path_cv}...")
    image_kit.save_to_file(output_path_cv)


    print("Все операции завершены!")
