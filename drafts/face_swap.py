import streamlit as st
from PIL import Image
import torch
import json
import os

# Импортируем функции из модулей
from modules.face_detection import detect_face
from modules.face_composite import composite_face
from modules.masking import create_face_mask
from modules.inpainting import run_inpainting
from modules.training import train_textual_inversion

def load_model_options(file_path="models.json"):
    """
    Загружает список доступных моделей из JSON-файла.
    Если файл не найден, возвращает значения по умолчанию.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                options = json.load(f)
            return options
        except Exception as e:
            st.warning(f"Ошибка при чтении {file_path}: {e}. Будут использованы значения по умолчанию.")
    # Значения по умолчанию
    return {
        "Stable Diffusion Inpainting": "runwayml/stable-diffusion-inpainting",
        "Stable Diffusion XL Inpainting": "stabilityai/stable-diffusion-xl-inpaint",
        "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1"
    }

def main():
    st.title("Face Swap with Diffusion Models")
    st.sidebar.header("Настройки приложения")
    mode = st.sidebar.radio("Режим работы", options=["Inference", "Training"])
    
    # Загружаем список моделей из файла (или используем значения по умолчанию)
    model_options = load_model_options()
    
    # Отображаем список доступных моделей в боковой панели
    with st.sidebar.expander("Доступные модели"):
        for name, repo in model_options.items():
            st.write(f"**{name}**: `{repo}`")
    
    # Выбор модели
    model_name = st.sidebar.selectbox("Выберите модель", options=list(model_options.keys()))
    selected_model = model_options[model_name]
    st.sidebar.info(f"Выбрана модель: **{model_name}**")
    
    if mode == "Inference":
        st.header("Инференс: Замена лица")
        st.write("Загрузите таргетное изображение и изображение референсного лица.")
        target_file = st.file_uploader("Таргетное изображение", type=["jpg", "jpeg", "png"], key="target")
        ref_file = st.file_uploader("Изображение референсного лица", type=["jpg", "jpeg", "png"], key="reference")
        
        strength = st.sidebar.slider("Сила преобразования", 0.0, 1.0, 0.7)
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.5)
        steps = st.sidebar.number_input("Шаги денойзинга", min_value=1, max_value=100, value=50)
        padding_mask_crop = st.sidebar.number_input("Padding mask crop (опционально)", min_value=0, value=0)
        
        if st.button("Запустить Inference"):
            if target_file is None or ref_file is None:
                st.error("Пожалуйста, загрузите оба изображения!")
            else:
                target_img = Image.open(target_file).convert("RGB")
                ref_img = Image.open(ref_file).convert("RGB")
                
                # Детекция лиц для вывода распознанных областей
                target_face_coords, _ = detect_face(target_img)
                ref_face_coords, _ = detect_face(ref_img)
                
                if target_face_coords is not None and ref_face_coords is not None:
                    x, y, w, h = target_face_coords
                    target_face_crop = target_img.crop((x, y, x + w, y + h))
                    x2, y2, w2, h2 = ref_face_coords
                    ref_face_crop = ref_img.crop((x2, y2, x2 + w2, y2 + h2))
                    
                    st.subheader("Распознанные лица")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(target_face_crop, caption="Распознанное лицо таргета", use_column_width=True)
                    with col2:
                        st.image(ref_face_crop, caption="Распознанное лицо референса", use_column_width=True)
                else:
                    st.warning("Не удалось распознать лицо на одном из изображений.")
                
                with st.spinner("Выполняется inpainting..."):
                    try:
                        result_img = run_inpainting(
                            target_img, ref_img,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            steps=steps,
                            padding_mask_crop=padding_mask_crop if padding_mask_crop > 0 else None,
                            model_name=selected_model
                        )
                    except Exception as e:
                        st.error(f"Ошибка при запуске модели: {e}")
                        return
                if result_img:
                    st.subheader("Сравнение до и после")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(target_img, caption="Таргет до изменения", use_column_width=True)
                    with col4:
                        st.image(result_img, caption="Результат inpainting", use_column_width=True)
    
    elif mode == "Training":
        st.header("Обучение модели")
        st.write("Настройте параметры обучения (например, для textual inversion).")
        training_files = st.file_uploader("Загрузите обучающие изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="train")
        new_token = st.text_input("Новый токен для обучения", value="mytoken")
        lr = st.number_input("Learning rate", value=5e-4, format="%.6f")
        train_steps = st.number_input("Шаги обучения", min_value=1, max_value=10000, value=1000)
        batch_size = st.number_input("Размер батча", min_value=1, max_value=16, value=1)
        
        if st.button("Начать обучение"):
            if not training_files:
                st.error("Пожалуйста, загрузите обучающие изображения!")
            else:
                training_images = [Image.open(f).convert("RGB") for f in training_files]
                with st.spinner("Обучение..."):
                    train_textual_inversion(training_images, new_token, lr, train_steps, batch_size)
                st.success("Обучение завершено (заглушка).")

if __name__ == "__main__":
    main()
