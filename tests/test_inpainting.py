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
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",
        "stabilityai/stable-diffusion-2-inpainting",
        "saik0s/realistic_vision_inpainting",
        "Lykon/dreamshaper-8-inpainting"
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
    {"strength": 0.3, "guidance_scale": 7.5, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.2},
    {"strength": 0.7, "guidance_scale": 12.0, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.2},
    {"strength": 0.75, "guidance_scale": 7.5, "use_blending": False, "use_face_restoration": False, "mask_scale_factor": 1.2},
    {"strength": 0.9, "guidance_scale": 10.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.7, "mask_scale_factor": 1.5},
    {"strength": 1.0, "guidance_scale": 6.0, "use_blending": False, "use_face_restoration": True, "restoration_strength": 0.5, "mask_scale_factor": 1.3},
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


if __name__ == "__main__":
    # Запускаем pytest
    pytest.main(["-v", __file__])
