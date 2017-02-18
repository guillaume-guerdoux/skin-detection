from skin_detector import ExplicitSkinDetector
from skin_detector import NonParametricSkinDetector
from skin_detector import Judge
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
        cv2.destroyAllWindows()'''
    learning_folders = [('data/Pratheepan_Dataset/FacePhoto/',
                         'data/Ground_Truth/GroundT_FacePhoto/')]
    non_parametric_skin_detector = NonParametricSkinDetector(learning_folders)
    # explicit_skin_detector = ExplicitSkinDetector()
    judge = Judge(explicit_skin_detector, learning_folders)
    judge.get_recall_precision()
    '''img = load_image('data/Pratheepan_Dataset/FacePhoto/920480_f520.jpg', True)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    skin_img = non_parametric_skin_detector.draw_skin(
        img, (255, 255, 255), (0, 0, 0)
    )
    cv2.imshow('image', skin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
