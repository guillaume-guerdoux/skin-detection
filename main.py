from skin_detector import ExplicitSkinDetector
from skin_detector import NonParametricBGRSkinDetector
from skin_detector import NonParametricHSVSkinDetector
from skin_detector import NonParametricLABSkinDetector
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
    test_folders = [('data/Pratheepan_Dataset/FamilyPhoto/',
                     'data/Ground_Truth/GroundT_FamilyPhoto/')]
    non_parametric_skin_detector = NonParametricBGRSkinDetector(learning_folders)
    # non_parametric_skin_detector = NonParametricHSVSkinDetector(learning_folders)
    # non_parametric_skin_detector = NonParametricLABSkinDetector(learning_folders)
    # explicit_skin_detector = ExplicitSkinDetector()
    judge = Judge(non_parametric_skin_detector, learning_folders)
    judge.get_recall_precision()
    '''img = load_image('data/tabatha.jpg', True)
    hsv_img = BGR_to_Lab(img)
    cv2.imshow('image', hsv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    skin_img = non_parametric_skin_detector.draw_skin(
        hsv_img, (255, 255, 255), (0, 0, 0)
    )
    cv2.imshow('image', skin_img)
    # cv2.imwrite('tabatha_skin_2.jpg', skin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
