import cv2
import numpy as np
from imutils import face_utils

def create_face_mask(image_shape, face_coords):
    """
    Создаёт маску для области лица в виде эллипса с плавным размытием.
    image_shape: форма исходного изображения (в формате cv2, например, (h, w, 3))
    face_coords: кортеж (x, y, w, h)
    Возвращает маску в формате numpy.ndarray с размерами (h, w).
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    (x, y, w, h) = face_coords
    center = (x + w // 2, y + h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask

def create_expanded_face_mask(image_shape, landmarks, scale_factor=1.1, blur_kernel=(15, 15)):
    """
    Создает маску для лица на основе выпуклой оболочки landmark'ов,
    расширяя ее на указанный коэффициент для охвата всего лица.
    
    :param image_shape: форма исходного изображения (например, (h, w, 3))
    :param landmarks: numpy-массив координат (N, 2) landmark'ов
    :param scale_factor: коэффициент масштабирования выпуклой оболочки (1.0 – без расширения)
    :param blur_kernel: размер ядра для гауссова размытия
    :return: маска в формате numpy.ndarray (значения от 0 до 255)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Вычисляем выпуклую оболочку
    hull = cv2.convexHull(landmarks)
    
    # Вычисляем центр выпуклой оболочки
    M = cv2.moments(hull)
    if M["m00"] == 0:
        center = np.mean(landmarks, axis=0)
    else:
        center = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
    
    # Расширяем точки выпуклой оболочки
    expanded_hull = []
    for point in hull:
        pt = point[0]
        vector = pt - center
        expanded_pt = center + scale_factor * vector
        expanded_hull.append(expanded_pt.astype(np.int32))
    expanded_hull = np.array(expanded_hull)
    
    # Заполняем маску расширенной оболочкой
    cv2.fillConvexPoly(mask, expanded_hull, 255)
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return mask
