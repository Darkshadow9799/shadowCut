import os, sys
from PIL import Image

def resize_image(path, new_path, resizing_params=(256,144), quality=90, type = 'JPEG'):
    directories = os.listdir(path, new_path, resizing_params)
    index = 1
    for directory in directories:
        if os.path.isfile(path + directory):
            image = Image.open(path + directory)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            _, e = os.path.splitext(path + directory)
            file_destination = new_path
            image_resize = image.resize(resizing_params, Image.ANTIALIAS)
            image_resize.save(file_destination + str(index) + e, type, quality = quality)
            index += 1