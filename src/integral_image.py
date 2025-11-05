import cv2
import numpy as np

def integral_image(img):
    ii = cv2.integral(img)[1:, 1:]
    return ii
