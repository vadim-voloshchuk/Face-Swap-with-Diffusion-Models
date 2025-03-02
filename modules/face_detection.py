import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
# import mediapipe as mp
import cv2

# Инициализация MTCNN
mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")

# Инициализация Mediapipe Face Mesh
# mp_face_detection = mp.solutions.face_detection
# face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_face(image: Image.Image, method="mtcnn"):
    """
    Детектирует лицо на изображении.

    Параметры:
    - image (PIL.Image): входное изображение.
    - method (str): метод детекции ("mtcnn" или "mediapipe").

    Возвращает:
    - (x, y, w, h): координаты лица или None, если лицо не найдено.
    - исходное изображение в формате OpenCV.
    """
    image_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    if method == "mtcnn":
        boxes, probs = mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, image_cv
        idx = np.argmax(probs)  # Берём самое уверенное лицо
        x, y, x2, y2 = boxes[idx].astype(int)
        return (x, y, x2 - x, y2 - y), image_cv

    # elif method == "mediapipe":
    #     results = face_detector.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    #     if results.detections:
    #         detection = results.detections[0]
    #         bboxC = detection.location_data.relative_bounding_box
    #         h, w, _ = image_cv.shape
    #         x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
    #         return (x, y, w, h), image_cv

    return None, image_cv
