from PIL import Image, ImageOps
import cv2
import numpy as np
import json
import torch
from diffusers import AutoPipelineForInpainting
from modules.face_detection import detect_face
from modules.face_composite import composite_face_aligned
from modules.masking import create_expanded_face_mask
from modules.face_restoration import restore_faces

# Дополнительные импорты для landmark conditioning
import dlib
from imutils import face_utils

# Функции для динамического паддинга
def pad_to_multiple(image, multiple=8):
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    padded_image = Image.new("RGB", (new_w, new_h))
    padded_image.paste(image, (0, 0))
    return padded_image, (w, h)

def crop_to_original(padded_image, orig_size):
    return padded_image.crop((0, 0, orig_size[0], orig_size[1]))

# Функции для многополосного blending-а (laplacian blending)
def gaussian_pyramid(image, levels=5):
    gp = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp

def laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gp[i+1], dstsize=size)
        laplacian = cv2.subtract(gp[i], gaussian_expanded)
        lp.append(laplacian)
    lp.append(gp[-1])
    return lp

def blend_pyramids(lpA, lpB, gp_mask):
    blended = []
    for la, lb, mask in zip(lpA, lpB, gp_mask):
        blended.append(la * mask + lb * (1 - mask))
    return blended

def reconstruct_from_pyramid(lp):
    image = lp[-1]
    for l in reversed(lp[:-1]):
        size = (l.shape[1], l.shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, l)
    return image

def laplacian_blending(generated, composite, mask, levels=5):
    gp_gen = gaussian_pyramid(generated, levels)
    gp_comp = gaussian_pyramid(composite, levels)
    gp_mask = gaussian_pyramid(mask, levels)
    
    lp_gen = laplacian_pyramid(gp_gen)
    lp_comp = laplacian_pyramid(gp_comp)
    
    blended_lp = blend_pyramids(lp_gen, lp_comp, gp_mask)
    blended = reconstruct_from_pyramid(blended_lp)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def load_prompts(prompt_file="config/prompts.json"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompt", ""), data.get("negative_prompt", "")

def run_inpainting(target_img, ref_img, strength, guidance_scale, steps, padding_mask_crop, model_name, 
                   prompt_file="config/prompts.json", use_blending=False, 
                   use_face_restoration=True, restoration_strength=0.5, mask_scale_factor=1.0):
    """
    Запускает inpainting с учётом landmarks и возможностью восстановления лица после генерации.
    Теперь ВСЕ модели проходят через AutoPipelineForInpainting.
    """
    # Загружаем промпты из внешнего файла
    prompt, negative_prompt = load_prompts(prompt_file)

    # Загружаем предсказатель landmark'ов
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    landmark_predictor = dlib.shape_predictor(predictor_path)
    
    # Детектируем лица на таргетном и референсном изображениях
    target_face_coords, target_cv = detect_face(target_img)
    ref_face_coords, ref_cv = detect_face(ref_img)
    if target_face_coords is None or ref_face_coords is None:
        raise ValueError("Не удалось обнаружить лицо на одном из изображений.")

    # Получаем landmarks целевого лица
    x, y, w, h = target_face_coords
    rect = dlib.rectangle(x, y, x + w, y + h)
    target_np = np.array(target_img.convert("RGB"))
    landmarks = landmark_predictor(target_np, rect)
    landmarks_np = face_utils.shape_to_np(landmarks)

    # Создаём расширенную маску лица
    mask_np = create_expanded_face_mask(target_np.shape, landmarks_np, scale_factor=mask_scale_factor, blur_kernel=(15, 15))
    
    # Композитинг: выравниваем референсное лицо на целевом
    composite_cv = composite_face_aligned(target_cv, ref_cv, predictor_path=predictor_path)

    # Преобразуем изображения в PIL
    composite_pil = Image.fromarray(cv2.cvtColor(composite_cv, cv2.COLOR_BGR2RGB)).convert("RGB")
    mask_pil = Image.fromarray(mask_np).convert("L")
    
    # Подгоняем размеры
    orig_size = composite_pil.size
    composite_padded, orig_size = pad_to_multiple(composite_pil, multiple=8)
    mask_padded, _ = pad_to_multiple(mask_pil, multiple=8)
    padded_w, padded_h = composite_padded.size

    # Загружаем AutoPipelineForInpainting для всех моделей
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Убираем variant
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели {model_name}: {e}")


    # Запуск inpainting
    result_padded = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=composite_padded,
        mask_image=mask_padded,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=padded_h,
        width=padded_w,
        padding_mask_crop=padding_mask_crop if padding_mask_crop and padding_mask_crop > 0 else None,
    ).images[0]

    result = crop_to_original(result_padded, orig_size)

    # Пост-обработка (Blending + Face Restoration)
    if use_blending:
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR).astype(np.float32)
        composite_np = cv2.cvtColor(np.array(composite_pil), cv2.COLOR_RGB2BGR).astype(np.float32)
        mask_np_float = np.array(mask_pil).astype(np.float32) / 255.0
        if len(mask_np_float.shape) == 2:
            mask_np_float = cv2.merge([mask_np_float, mask_np_float, mask_np_float])
        
        blended_np = laplacian_blending(result_np, composite_np, mask_np_float, levels=5)
        result = Image.fromarray(cv2.cvtColor(blended_np, cv2.COLOR_BGR2RGB))

    if use_face_restoration:
        result = restore_faces(result, fidelity=restoration_strength)

    return result
