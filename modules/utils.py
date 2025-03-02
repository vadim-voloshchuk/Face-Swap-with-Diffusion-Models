from PIL import Image, ImageOps

def pad_to_multiple_bottom_right(image: Image.Image, multiple=8):
    """
    Добавляет паддинг только справа и снизу, чтобы размеры изображения стали кратными multiple.
    Возвращает:
      - padded_image: PIL.Image с добавленным паддингом,
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
    Обрезает изображение, возвращая его до исходного размера.
    """
    orig_w, orig_h = orig_size
    return image.crop((0, 0, orig_w, orig_h))
