import os
import pytest
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline
from modules.inpainting import run_inpainting

# 🔹 Получаем все модели, содержащие "inpaint" в названии (из diffusers)
def get_available_inpaint_models():
    """Сканирует diffusers и возвращает список доступных моделей для inpainting."""
    all_models = [
        "runwayml/stable-diffusion-inpainting",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    ]
    
    available_models = []
    for model in all_models:
        try:
            # Проверяем, может ли pipeline загрузить модель
            pipe = DiffusionPipeline.from_pretrained(model)
            if "inpaint" in model.lower():
                available_models.append(model)
        except:
            print(f"⚠️ Модель {model} не загрузилась.")
    
    return available_models

# 🔹 Получаем список всех доступных моделей
MODEL_LIST = get_available_inpaint_models()

# 🔹 Разные параметры для тестирования
PARAMS_LIST = [
    {"strength": 0.6, "guidance_scale": 5.0, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.0},
    {"strength": 0.8, "guidance_scale": 7.5, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.2},
    {"strength": 0.9, "guidance_scale": 10.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.7, "mask_scale_factor": 1.5},
    {"strength": 0.7, "guidance_scale": 6.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.5, "mask_scale_factor": 1.3},
    {"strength": 0.85, "guidance_scale": 8.0, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.4}
]

# 🔹 Пути к тестовым изображениям
TARGET_IMAGE_PATH = "data/target/IMG_9240.jpg"
REFERENCE_IMAGE_PATH = "data/glam_test_task_data (2)/IMG_0738.jpg"

# 🔹 Директория для сохранения результатов
RESULTS_DIR = "tests/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@pytest.mark.parametrize("model_name", MODEL_LIST)
@pytest.mark.parametrize("params", PARAMS_LIST)
def test_inpainting(model_name, params):
    """Тестирует inpainting для каждой модели с разными параметрами и сохраняет результаты."""
    
    target_img = Image.open(TARGET_IMAGE_PATH).convert("RGB")
    ref_img = Image.open(REFERENCE_IMAGE_PATH).convert("RGB")
    
    # Создаём папку для модели
    model_dir = os.path.join(RESULTS_DIR, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    # Запуск inpainting
    result = run_inpainting(
        target_img, ref_img, model_name=model_name, steps=20, padding_mask_crop=0, **params
    )

    if isinstance(result, np.ndarray):
        result = Image.fromarray(result)
        
    assert result is not None, f"Inpainting не вернул изображение (модель: {model_name})"
    
    # Сохраняем результат
    filename = f"s{params['strength']}_g{params['guidance_scale']}.jpg"
    result_path = os.path.join(model_dir, filename)
    result.save(result_path)
    print(f"✅ Сохранён результат: {result_path}")

#!/usr/bin/env python3
import json
from huggingface_hub import HfApi
from PIL import Image, ImageDraw, ImageFont
import os

def main():
    api = HfApi()
    
    # Получаем список моделей с фильтрацией по text-to-image
    models = list(api.list_models(filter="text-to-image", sort="downloads", direction=-1))
    
    # Фильтруем только inpaint-модели (проверяем по названию или тегам)
    inpaint_models = [
        model for model in models 
        if "inpaint" in model.modelId.lower() or (model.tags and "inpainting" in model.tags)
    ]
    
    # Берем топ-5 моделей по загрузкам
    top5 = inpaint_models[:5]
    
    # Формируем словарь, где ключи - названия моделей, а значения - идентификаторы
    model_dict = {model.modelId.split("/")[-1]: model.modelId for model in top5}
    
    # Сохраняем результаты в JSON-файл
    with open("inpaint_models.json", "w", encoding="utf-8") as f:
        json.dump(model_dict, f, ensure_ascii=False, indent=4)
    
    print("Топ-5 inpaint diffusion моделей сохранены в inpaint_models.json")
    
PARAMS_LIST = [
    {"strength": 0.6, "guidance_scale": 5.0, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.0},
    {"strength": 0.8, "guidance_scale": 7.5, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.2},
    {"strength": 0.9, "guidance_scale": 10.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.7, "mask_scale_factor": 1.5},
    {"strength": 0.7, "guidance_scale": 6.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.5, "mask_scale_factor": 1.3},
    {"strength": 0.85, "guidance_scale": 8.0, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.4}
]

def create_comparison_images():
    """Создаёт отдельные сравнительные изображения для каждой модели."""
    FONT_SIZE = 48
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)  # complex шрифт
    except:
        font = ImageFont.load_default(size=FONT_SIZE)
    
    for model in MODEL_LIST:
        model_dir = os.path.join(RESULTS_DIR, model.replace("/", "_"))
        result_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".jpg")])
        
        if not result_files:
            print(f"❌ Нет результатов для {model}!")
            continue
        
        # Определяем минимальный размер среди всех изображений
        min_w, min_h = float('inf'), float('inf')
        images = {}
        for filename in result_files:
            img = Image.open(os.path.join(model_dir, filename))
            images[filename] = img
            min_w, min_h = min(min_w, img.width), min(min_h, img.height)

        # Масштабируем изображения под минимальный размер с сохранением пропорций
        for filename, img in images.items():
            img.thumbnail((min_w, min_h), Image.LANCZOS)
            images[filename] = img

        # Отступы рассчитываем относительно размера шрифта
        spacing = int(FONT_SIZE * 3)
        title_height = int(FONT_SIZE * 10)
        label_offset = int(FONT_SIZE * 6)


        # Создаём изображение с подписями
        cols = len(PARAMS_LIST)
        big_img_w = cols * min_w + (cols + 1) * spacing
        big_img_h = min_h + title_height + label_offset

        big_image = Image.new("RGB", (big_img_w, big_img_h), "white")
        draw = ImageDraw.Draw(big_image)

        # Заголовок
        title_text = f"Сравнение модели: {model}"
        draw.text((big_img_w // 2 - len(title_text) * 12, FONT_SIZE // 2), title_text, fill="black", font=font)

        # Вставляем изображения и подписываем их
        x_offset = spacing
        y_offset = title_height
        for col_idx, param in enumerate(PARAMS_LIST):
            filename = f"s{param['strength']}_g{param['guidance_scale']}.jpg"

            if filename in images:
                big_image.paste(images[filename], (x_offset, y_offset))
                
                param_text = (f"Strength: {param['strength']}, Guidance: {param['guidance_scale']}\n"
                              f"Face Restoration: {param['use_face_restoration']}, "
                              f"Restoration Strength: {param.get('restoration_strength', 'N/A')}\n"
                              f"Mask Scale Factor: {param['mask_scale_factor']}")
                draw.text((x_offset + 10, y_offset - label_offset), param_text, fill="black", font=font)
            
            x_offset += min_w + spacing

        # Сохранение сравнительного изображения
        comparison_path = os.path.join(model_dir, "comparison.jpg")
        big_image.save(comparison_path)
        print(f"📊 Сравнение сохранено в {comparison_path}")

create_comparison_images()

if __name__ == "__main__":
    # Запускаем pytest
    pytest.main(["-v", __file__])
    
    # Генерируем сравнения
    create_comparison_images()
