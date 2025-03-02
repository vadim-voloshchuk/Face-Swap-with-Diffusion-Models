import cv2
import dlib
import numpy as np
from imutils import face_utils

def get_landmarks(image, detector, predictor, upsample_times=1):
    """
    Определяет координаты 68 landmark точек на лице.
    Возвращает np.array с координатами или None, если лицо не найдено.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, upsample_times)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    coords = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)], dtype=np.float32)
    return coords

def align_face(ref_image, target_image, detector, predictor, upsample_ref=1, upsample_target=1):
    """
    Выравнивает референсное изображение по геометрии целевого.
    Возвращает выровненное референсное изображение и матрицу преобразования.
    """
    ref_landmarks = get_landmarks(ref_image, detector, predictor, upsample_ref)
    target_landmarks = get_landmarks(target_image, detector, predictor, upsample_target)
    if ref_landmarks is None or target_landmarks is None:
        raise ValueError("Не удалось обнаружить лицо на одном из изображений.")
    
    # Используем три ключевые точки: центры глаз и точку носа
    left_eye_ref = np.mean(ref_landmarks[36:42], axis=0)
    right_eye_ref = np.mean(ref_landmarks[42:48], axis=0)
    nose_ref = ref_landmarks[30]

    left_eye_target = np.mean(target_landmarks[36:42], axis=0)
    right_eye_target = np.mean(target_landmarks[42:48], axis=0)
    nose_target = target_landmarks[30]

    src_points = np.array([left_eye_ref, right_eye_ref, nose_ref], dtype=np.float32)
    dst_points = np.array([left_eye_target, right_eye_target, nose_target], dtype=np.float32)

    M = cv2.getAffineTransform(src_points, dst_points)
    h, w = target_image.shape[:2]
    aligned_ref = cv2.warpAffine(ref_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned_ref, M

def create_expanded_face_mask(image_shape, landmarks, scale_factor=1.2, blur_kernel=(21,21)):
    """
    Создаёт маску лица на основе выпуклой оболочки landmark'ов,
    расширяя её на scale_factor для полного охвата лица и смягчения границ.
    
    :param image_shape: форма исходного изображения (например, (h, w, 3))
    :param landmarks: numpy-массив координат (N, 2) landmark'ов
    :param scale_factor: коэффициент масштабирования выпуклой оболочки (1.0 – без расширения)
    :param blur_kernel: размер ядра для гауссова размытия
    :return: маска в формате numpy.ndarray (0-255)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(landmarks)
    
    # Находим центр лица
    M = cv2.moments(hull)
    if M["m00"] != 0:
        center = np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
    else:
        center = np.mean(landmarks, axis=0)
    
    # Расширяем каждую точку оболочки относительно центра
    expanded_hull = []
    for point in hull:
        pt = point[0]
        vector = pt - center
        expanded_pt = center + scale_factor * vector
        expanded_hull.append(expanded_pt.astype(np.int32))
    expanded_hull = np.array(expanded_hull)
    
    cv2.fillConvexPoly(mask, expanded_hull, 255)
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return mask

def composite_face_aligned(target_image, ref_image, predictor_path="shape_predictor_68_face_landmarks.dat", scale_factor=1.2):
    """
    Выравнивает лицо из референсного изображения по геометрии целевого лица,
    создаёт маску на основе landmark'ов, расширенную с коэффициентом scale_factor,
    и выполняет seamlessClone для интеграции выровненного лица в исходное изображение.
    
    :param target_image: исходное изображение (numpy.ndarray, BGR)
    :param ref_image: референсное изображение (numpy.ndarray, BGR)
    :param predictor_path: путь к файлу модели dlib
    :param scale_factor: коэффициент расширения маски
    :return: композитное изображение (numpy.ndarray, BGR)
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Вычисляем landmarks для целевого лица
    target_landmarks = get_landmarks(target_image, detector, predictor)
    if target_landmarks is None:
        raise ValueError("Не удалось обнаружить лицо на целевом изображении.")

    # Вычисляем маску на основе landmark'ов
    mask = create_expanded_face_mask(target_image.shape, target_landmarks, scale_factor=scale_factor, blur_kernel=(21,21))
    
    # Выравниваем референсное изображение по целевому
    aligned_ref, _ = align_face(ref_image, target_image, detector, predictor)
    
    # Определяем центр маски для seamlessClone
    x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(mask)
    center = (x_mask + w_mask // 2, y_mask + h_mask // 2)
    
    # Композитинг с помощью seamlessClone
    composite = cv2.seamlessClone(aligned_ref, target_image, mask, center, cv2.NORMAL_CLONE)
    return composite

# Пример использования:
if __name__ == "__main__":
    ref_path = "reference.jpg"
    target_path = "target.jpg"
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    ref_image = cv2.imread(ref_path)
    target_image = cv2.imread(target_path)
    
    if ref_image is None or target_image is None:
        raise FileNotFoundError("Не удалось загрузить одно из изображений.")
    
    result = composite_face_aligned(target_image, ref_image, predictor_path, scale_factor=1.2)
    
    cv2.imshow("Composite Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
