import cv2
import numpy as np

def normalize_image(img):
    # Normalize by standard deviation
    means = np.mean(img, axis=(0, 1))
    means = means[None,:]

    std = np.std(img, axis=(0, 1))
    std = std[None,:]
    return (img - means) / std

def preprocess_image(img):
    # Original image is (160, 320, 3)
    img_crop = img[60:150, :, :] # Clip mainly the top of the image (94, 320, 3)
    img_resize = cv2.resize(img_crop, (200, 66)) 
    img_normed = normalize_image(img_resize)
    return img_normed