import os
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import normalize
from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils import img2tensor, tensor2img
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.utils.misc import is_gray

# Пути к предобученным моделям
pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}

# Проверяем и загружаем недостающие веса
weights_dir = "CodeFormer/weights"
os.makedirs(weights_dir, exist_ok=True)

for key, url in pretrain_model_url.items():
    model_path = os.path.join(weights_dir, key, os.path.basename(url))
    if not os.path.exists(model_path):
        load_file_from_url(url=url, model_dir=os.path.join(weights_dir, key), progress=True, file_name=None)

# Инициализируем модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация RealESRGAN для повышения качества изображений
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path=os.path.join(weights_dir, "realesrgan", "RealESRGAN_x2plus.pth"),
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

# Загружаем CodeFormer
upsampler = set_realesrgan()
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)

# Загружаем веса модели
ckpt_path = os.path.join(weights_dir, "codeformer", "codeformer.pth")
checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

def restore_faces(image, background_enhance=True, face_upsample=True, upscale=2, fidelity=0.5):
    """
    Восстанавливает лицо на изображении с помощью CodeFormer.
    
    :param image: Входное изображение (PIL или numpy).
    :param background_enhance: Улучшать ли фон (используя RealESRGAN).
    :param face_upsample: Улучшать ли сами лица.
    :param upscale: Коэффициент апскейлинга.
    :param fidelity: Степень реставрации (0 — агрессивная генерация, 1 — сохранение оригинального лица).
    :return: Восстановленное изображение (numpy).
    """
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)

    upscale = min(max(int(upscale), 1), 4)  # Ограничиваем значения апскейлинга
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )

    bg_upsampler = upsampler if background_enhance else None
    face_upsampler = upsampler if face_upsample else None

    face_helper.read_image(img)
    num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)

    print("Начинаем считать")
    if num_faces == 0:
        return img  # Если лиц не найдено, возвращаем оригинал

    # Восстановление каждого лица
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        with torch.no_grad():
            output = codeformer_net(cropped_face_t, w=fidelity, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        face_helper.add_restored_face(restored_face.astype("uint8"))

    print("Посчитали")
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(
        upsample_img=bg_upsampler.enhance(img, outscale=upscale)[0] if bg_upsampler else None,
        draw_box=False,
        face_upsampler=face_upsampler,
    )

    print("Возвращаем")
    
    return restored_img
