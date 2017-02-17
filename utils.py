import numpy as np
from matplotlib import pyplot as plt
import cv2


def load_image(nom, is_color):
    # Channel BGR
    if is_color:
        img = cv2.imread(nom, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(nom, cv2.IMREAD_GRAYSCALE)
    return img


if __name__ == '__main__':
    img = load_image("data/lenna.jpg", True)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
