import cv2
import streamlit as st
from PIL import Image
import numpy as np

from ocr.ocr_metrics import OcrMetrics
from ocr.ocr_model import EasyOcrModel
from pre_process.cv_preprocess import ImgPreprocessing
from utils.image_utils import ImageKit


def main(img_util, ocr_model, ocr_metrics, img_process):

    st.title("Приложение для обработки изображений")

    # Создание боковой панели
    with st.sidebar:
        st.header("Параметры обработки")

        # Загрузка изображения
        uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

        # Ползунки для регулировки параметров
        brightness = st.slider("Яркость", -1.0, 1.0, 0.50, step=0.1)
        contrast = st.slider("Контраст", 0.1, 4.0, 1.0, step=0.1)
        scale = st.slider("Масштаб", 0.5, 4.0, 1.0, step=0.5)

        # Чекбоксы для автоповорота и порога
        threshold = st.checkbox("Включить пороговую обработку")
        auto_rotate = st.checkbox("Включить автоповорот")

    # Основная область
    col1, col2 = st.columns(2)

    # Отображение изображений и текстовых полей
    with col1:
        st.header("Исходное изображение")
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption='Исходное изображение', use_container_width=True)
            st.text_area("Описание исходного изображения", height=100)
        else:
            st.text("Пожалуйста, загрузите изображение.")

    with col2:
        st.header("Обработанное изображение")
        if uploaded_file is not None:
            # загрузка изображения для обработки в формате cv
            rgb_image = np.array(original_image)
            image_cv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            img_process.image_fromarray(image_cv)
            # обработка изображения с учетом параметров
            processed_image = img_process.run(
                rotate=auto_rotate, rotation_angle=None,
                brightness_factor=brightness,
                contrast_factor=contrast,
                scale_factor=scale,
                thresholding=threshold
            )
            st.image(processed_image, caption='Обработанное изображение', use_container_width=True)
            st.text_area("Описание обработанного изображения", height=100)
        else:
            st.text("Пожалуйста, загрузите изображение.")


if __name__ == "__main__":
    image_kit = ImageKit()
    ocr_easy = EasyOcrModel()
    ocr_wer = OcrMetrics()
    image_processor = ImgPreprocessing()
    main(image_kit, ocr_easy, ocr_wer, image_processor)