from skin_detector import ExplicitSkinDetector
from skin_detector import NonParametricSkinDetector
from utils import *

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':
    '''explicit_skin_detector = ExplicitSkinDetector()
    folder = "data/Pratheepan_Dataset/FacePhoto/"
    for filename in os.listdir(folder):
        img = load_image(folder + filename, True)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        skin_img = explicit_skin_detector.draw_skin(
            img, (255, 255, 255), (0, 0, 0)
        )
        cv2.imshow('image', skin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img = load_image('data/lenna.jpg', True)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    skin_img = explicit_skin_detector.draw_skin(
        img, (255, 255, 255), (0, 0, 0)
    )
    cv2.imshow('image', skin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    skin_detector = NonParametricSkinDetector()
    skin_detector.create_skin_models([("data/Pratheepan_Dataset/FacePhoto/",
                                      "data/Ground_Truth/GroundT_FacePhoto/")])
