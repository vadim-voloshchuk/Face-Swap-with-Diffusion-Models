import os
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline

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
def create_comparison_images():
    """Создаёт отдельные сравнительные изображения для каждой модели."""
    FONT_SIZE = 48
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)  # попытка загрузить сложный шрифт
    except Exception:
        font = ImageFont.load_default(size=FONT_SIZE)

    for model in MODEL_LIST:
        model_dir = os.path.join(RESULTS_DIR, model.replace("/", "_"))
        if not os.path.isdir(model_dir):
            print(f"❌ Директория {model_dir} не найдена для модели {model}!")
            continue

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

        # Масштабируем изображения до минимального размера с сохранением пропорций
        for filename, img in images.items():
            img.thumbnail((min_w, min_h), Image.LANCZOS)
            images[filename] = img

        # Расчёт отступов относительно размера шрифта
        spacing = int(FONT_SIZE * 3)
        title_height = int(FONT_SIZE * 10)
        label_offset = int(FONT_SIZE * 6)

        # Размер итогового изображения
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
        for param in PARAMS_LIST:
            filename = f"s{param['strength']}_g{param['guidance_scale']}.jpg"
            if filename in images:
                big_image.paste(images[filename], (x_offset, y_offset))
                param_text = (
                    f"Strength: {param['strength']}, Guidance: {param['guidance_scale']}\n"
                    f"Face Restoration: {param['use_face_restoration']}, "
                    f"Restoration Strength: {param.get('restoration_strength', 'N/A')}\n"
                    f"Mask Scale Factor: {param['mask_scale_factor']}"
                )
                draw.text((x_offset + 10, y_offset - label_offset), param_text, fill="black", font=font)
            else:
                print(f"⚠️ Изображение {filename} не найдено в {model_dir}!")
            x_offset += min_w + spacing

        # Сохранение сравнительного изображения
        comparison_path = os.path.join(model_dir, "comparison.jpg")
        big_image.save(comparison_path)
        print(f"📊 Сравнение сохранено в {comparison_path}")

if __name__ == '__main__':
    create_comparison_images()
