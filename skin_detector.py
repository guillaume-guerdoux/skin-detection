import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
from utils import *


class Judge:

    def __init__(self, skin_detector, tests_folders):
        '''skin_detector : Object SkinDetector
        folders = list of tuple :
            [(folder1, folderstrue),(folder2, folder2true...)]'''
        self.skin_detector = skin_detector
        self.tests_folders = tests_folders

    def get_recall_precision(self):
        recalls = []
        precisions = []
        for test_folder in self.tests_folders:
            for filename in os.listdir(test_folder[0]):
                # print(filename)
                true_positive = 0
                false_positive = 0
                false_negative = 0
                true_negative = 0
                color_img = load_image(test_folder[0] + filename, True)
                black_and_white_img = load_image(
                    test_folder[1] + os.path.splitext(filename)[0]+'.png',
                    True)
                for index_row, row in enumerate(color_img):
                    for index_triplet, triplet in enumerate(row):
                        black_and_white_color = \
                            black_and_white_img[index_row][index_triplet]
                        if list(black_and_white_color) == [0, 0, 0]:
                            if self.skin_detector.is_skin_pixel(triplet):
                                false_negative += 1
                            else:
                                true_negative += 1
                        elif list(black_and_white_color) == [255, 255, 255]:
                            if self.skin_detector.is_skin_pixel(triplet):
                                true_positive += 1
                            else:
                                false_positive += 1
                        else:
                            print("ok")
                recall = true_positive/(true_positive + false_positive)
                precision = true_positive/(true_positive + false_negative)
                # print('Recall', recall)
                # print("Precision", precision)
                recalls.append(recall)
                precisions.append(precision)
        print('Mean recall', (sum(recalls)/len(recalls)))
        print('Mean precision', (sum(precisions)/len(precisions)))


class SkinDetector:

    def draw_skin(self, img, color_for_skin, color_for_not_skin):
        # img is a cv image
        # color_for_skin : (B, G, R) to colorize skin pixels
        # color_for_not_skin : (B, G, R) to colorize not skin pixels
        for index_row, row in enumerate(img):
            for index_triplet, triplet in enumerate(row):
                if self.is_skin_pixel(triplet):
                    img[index_row][index_triplet] = color_for_skin
                else:
                    img[index_row][index_triplet] = color_for_not_skin
        return img


class ExplicitSkinDetector(SkinDetector):

    def is_skin_pixel(self, pixel):
        # pixel : (B, G, R)
        B = pixel[0]
        G = pixel[1]
        R = pixel[2]
        # Formula in assets/gc2003vsa.pdf article
        if R > 95 and G > 40 and B > 20 and \
           (max(R, G, B) - min(R, G, B)) > 15 and \
           abs(int(R) - int(G)) > 15 and R > G and R > B:
                return True
        else:
            return False


class NonParametricBGRSkinDetector(SkinDetector):

    def __init__(self, learning_folders):
        skin_models = self.create_skin_models(learning_folders)
        self.skin_model = skin_models[0]
        self.non_skin_model = skin_models[1]

    def create_skin_models(self, learning_folders):
        '''learning_folders : lists of differents sample learning_folders
        Two arrays dimension (a and b) : 0 -> 256 0 -> 256
        One array skin model : nb of skin pixel in image
        One array non skin model : nb of non skin pixel in image'''
        skin_model = np.zeros((32, 32))
        non_skin_model = np.zeros((32, 32))
        for folder in learning_folders:
            for filename in os.listdir(folder[0]):
                color_img = load_image(folder[0] + filename, True)
                black_and_white_img = load_image(
                    folder[1] + os.path.splitext(filename)[0]+'.png', True)
                inversed_black_and_white_img = \
                    inverse_image(black_and_white_img)
                temp_skin_histogram = BGR_histogram(
                    color_img, black_and_white_img
                )
                skin_model = np.add(skin_model, temp_skin_histogram)
                temp_non_skin_histogram = BGR_histogram(
                    color_img, inversed_black_and_white_img
                )
                non_skin_model = np.add(
                    non_skin_model, temp_non_skin_histogram
                )
        skin_model = np.divide(skin_model, sum(sum(skin_model)))
        non_skin_model = np.divide(non_skin_model, sum(sum(non_skin_model)))
        return skin_model, non_skin_model

    def is_skin_pixel(self, pixel):
        # pixel : (B, G, R)
        # 256 bins to 32 bins transformation
        B = int(pixel[0]/8)
        G = int(pixel[1]/8)
        R = int(pixel[2]/8)
        '''if R in self.red_list and G in self.green_list:
            print(R, G)'''
        p_skin = self.skin_model[R][G]
        p_non_skin = self.non_skin_model[R][G]
        if p_skin >= p_non_skin:
            return True
        else:
            return False


class NonParametricHSVSkinDetector(SkinDetector):

    def __init__(self, learning_folders):
        skin_models = self.create_skin_models(learning_folders)
        self.skin_model = skin_models[0]
        self.non_skin_model = skin_models[1]

    def create_skin_models(self, learning_folders):
        '''learning_folders : lists of differents sample learning_folders
        Two arrays dimension (a and b) : 0 -> 256 0 -> 256
        One array skin model : nb of skin pixel in image
        One array non skin model : nb of non skin pixel in image'''
        skin_model = np.zeros((180, 256))
        non_skin_model = np.zeros((180, 256))
        for folder in learning_folders:
            for filename in os.listdir(folder[0]):
                color_img = load_image(folder[0] + filename, True)
                hsv_img = BGR_to_HSV(color_img)
                black_and_white_img = load_image(
                    folder[1] + os.path.splitext(filename)[0]+'.png', True)
                inversed_black_and_white_img = \
                    inverse_image(black_and_white_img)
                temp_skin_histogram = HSV_histogram(
                    hsv_img, black_and_white_img
                )
                skin_model = np.add(skin_model, temp_skin_histogram)
                temp_non_skin_histogram = HSV_histogram(
                    hsv_img, inversed_black_and_white_img
                )
                non_skin_model = np.add(
                    non_skin_model, temp_non_skin_histogram
                )
        skin_model = np.divide(skin_model, sum(sum(skin_model)))
        non_skin_model = np.divide(non_skin_model, sum(sum(non_skin_model)))
        return skin_model, non_skin_model

    def is_skin_pixel(self, pixel):
        # pixel : (H, S, V)
        # 256 bins to 32 bins transformation
        H = pixel[0]
        S = pixel[1]
        V = pixel[2]
        p_skin = self.skin_model[H][S]
        p_non_skin = self.non_skin_model[H][S]
        if p_skin >= p_non_skin:
            return True
        else:
            return False
