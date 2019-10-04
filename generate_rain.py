import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage.color import gray2rgb
from skimage.util import random_noise


def gen_rain(img_path, img_size):
    # PARAMETER
    # motion blur
    SIZE = 30
    
    origin_image = Image.open(img_path)
    resized_image = origin_image.resize(img_size)
    
    # random noise
    black = np.zeros((img_size[1], img_size[0]), dtype=np.int8)
    black = random_noise(black, mode='s&p', salt_vs_pepper=0.3)
    black = 100 * black
    black = black.astype(np.uint8)
    black_img = Image.fromarray(black)
    
    # Gaussian Blur
    black_img = black_img.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    # Motion Blur
    kernel_motion_blur = np.zeros((SIZE, SIZE))
    
    kernel_motion_blur[:, int(SIZE-1/2)] = np.ones(SIZE)
    kernel_motion_blur = kernel_motion_blur / SIZE
    
    rain_layer = cv2.filter2D(np.array(black_img), -1, kernel_motion_blur)
    
    # synthesize
    rain_layer = gray2rgb(rain_layer)
    rain_img = np.minimum(np.array(resized_image, dtype=np.uint16) + rain_layer, 255)
    
    return rain_img, np.array(resized_image)


def save_img(numpy_img, path, filename):
    numpy_img = np.array(numpy_img, dtype=np.uint8)
    img = Image.fromarray(numpy_img)
    img.save(path + filename)