import cv2
import numpy as np
from PIL import Image, ImageOps
import argparse
import torch
import math
from diffusers import StableDiffusionInpaintPipeline

import torch

if torch.cuda.is_available():
    print("CUDA доступна!")
    print("Текущий GPU:", torch.cuda.current_device())
    print("Название GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA не доступна. Используется CPU.")


def detect_face(image_path):
    """
    Обнаруживает лицо на изображении по указанному пути с помощью OpenCV Haar Cascade.
    Возвращает координаты (x, y, w, h) и исходное изображение (BGR).
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("Лицо не обнаружено!")
        return None, image
    (x, y, w, h) = faces[0]
    return (x, y, w, h), image

def composite_face(target_image, target_face_coords, ref_image, ref_face_coords):
    """
    Вставляет лицо из референсного изображения в таргетное с помощью seamlessClone.
    """
    (x_t, y_t, w_t, h_t) = target_face_coords
    (x_r, y_r, w_r, h_r) = ref_face_coords
    ref_face = ref_image[y_r:y_r+h_r, x_r:x_r+w_r]
    ref_face_resized = cv2.resize(ref_face, (w_t, h_t))
    mask = 255 * np.ones(ref_face_resized.shape, ref_face_resized.dtype)
    center = (x_t + w_t // 2, y_t + h_t // 2)
    composite = cv2.seamlessClone(ref_face_resized, target_image, mask, center, cv2.NORMAL_CLONE)
    return composite

def create_face_mask(image_shape, face_coords):
    """
    Создаёт маску для области лица (эллипс с плавным размытием).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    (x, y, w, h) = face_coords
    center = (x + w // 2, y + h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask

def pad_to_multiple_bottom_right(image: Image.Image, multiple=8):
    """
    Добавляет паддинг только справа и снизу, чтобы размеры изображения стали кратными multiple.
    Возвращает:
      - padded_image: изображение с добавленным паддингом,
      - (pad_w, pad_h): сколько пикселей добавлено справа и снизу.
    """
    w, h = image.size
    new_w = (w + multiple - 1) // multiple * multiple
    new_h = (h + multiple - 1) // multiple * multiple
    pad_w = new_w - w
    pad_h = new_h - h
    if pad_w == 0 and pad_h == 0:
        return image, (0, 0)
    padded = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    return padded, (pad_w, pad_h)

def crop_from_pad_bottom_right(image: Image.Image, orig_size):
    """
    Обрезает изображение, возвращая исходный размер (без паддинга).
    """
    orig_w, orig_h = orig_size
    return image.crop((0, 0, orig_w, orig_h))

def main(args):
    # 1. Детектируем лицо на таргетном изображении
    target_face_coords, target_cv2 = detect_face(args.target)
    if target_face_coords is None:
        print("Ошибка обнаружения лица в таргетном изображении!")
        return

    # 2. Детектируем лицо на референсном изображении
    ref_face_coords, ref_cv2 = detect_face(args.reference)
    if ref_face_coords is None:
        print("Ошибка обнаружения лица в референсном изображении!")
        return

    # 3. Композитное наложение лица
    composite_cv2 = composite_face(target_cv2, target_face_coords, ref_cv2, ref_face_coords)

    # 4. Создаём маску для области лица
    mask_np = create_face_mask(target_cv2.shape, target_face_coords)

    # 5. Переводим композитное изображение и маску в PIL.Image
    composite_rgb = cv2.cvtColor(composite_cv2, cv2.COLOR_BGR2RGB)
    composite_pil = Image.fromarray(composite_rgb).convert("RGB")
    mask_pil = Image.fromarray(mask_np).convert("L")

    # 6. Запоминаем исходный размер таргетного изображения (без изменений)
    orig_size = composite_pil.size  # (width, height)

    # 7. Добавляем паддинг справа и снизу до кратности 8 (без масштабирования и смещения в угол)
    composite_padded, pad_xy = pad_to_multiple_bottom_right(composite_pil, multiple=8)
    mask_padded, _ = pad_to_multiple_bottom_right(mask_pil, multiple=8)

    # 8. Получаем размеры после паддинга (они будут переданы в pipeline)
    padded_w, padded_h = composite_padded.size

    # 9. Настраиваем StableDiffusionInpaintPipeline с явным указанием размеров
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 10. Формируем prompt: заменяем лицо, оставляя фон неизменным
    prompt = (
        "A photorealistic portrait. Replace the face with the reference face while keeping the background unchanged. "
        "The inpainted face must be natural and well-integrated."
    )

    # 11. Запускаем inpainting с указанием размеров, чтобы модель не масштабировала изображение до 512x512
    result_padded = pipe(
        prompt=prompt,
        image=composite_padded,
        mask_image=mask_padded,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        height=padded_h,
        width=padded_w,
        padding_mask_crop=args.padding_mask_crop,  # можно оставить None, если не требуется crop внутри маски
    ).images[0]

    # 12. Обрезаем паддинг и возвращаем изображение до исходного размера
    result = crop_from_pad_bottom_right(result_padded, orig_size)
    result.save(args.output)
    print(f"Результат сохранён в {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Swap Inpainting preserving original proportions.")
    parser.add_argument("--target", type=str, required=True, help="Путь к таргетному изображению")
    parser.add_argument("--reference", type=str, required=True, help="Путь к изображению референсного лица")
    parser.add_argument("--output", type=str, default="output.png", help="Путь для сохранения результата")
    parser.add_argument("--strength", type=float, default=0.7, help="Сила преобразования (0-1)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale для prompt")
    parser.add_argument("--steps", type=int, default=20, help="Количество шагов денойзинга")
    parser.add_argument("--padding_mask_crop", type=int, default=None,
                        help="Размер дополнительного crop для маски, если требуется (по умолчанию None)")
    args = parser.parse_args()
    main(args)
