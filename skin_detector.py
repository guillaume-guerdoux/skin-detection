import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
from utils import *


class ExplicitSkinDetector:

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


class NonParametricSkinDetector:

    def create_skin_models(self, folders):
        '''folder : lists of differents sample folders
        Two arrays dimension (a and b) : 0 -> 256 0 -> 256
        One array skin model : nb of skin pixel in image
        One array non skin model : nb of non skin pixel in image'''
        skin_models = np.zeros((32, 32))
        non_skin_models = np.zeros((32, 32))
        for folder in folders:
            for filename in os.listdir(folder[0]):
                color_img = load_image(folder[0] + filename, True)
                black_and_white_img = load_image(
                    folder[1] + os.path.splitext(filename)[0]+'.png', True)
                inversed_black_and_white_img = \
                    inverse_image(black_and_white_img)
                temp_skin_histogram = BGR_histogram(
                    color_img, black_and_white_img
                )
                skin_models = np.add(skin_models, temp_skin_histogram)
                temp_non_skin_histogram = BGR_histogram(
                    color_img, inversed_black_and_white_img
                )
                non_skin_models = np.add(
                    non_skin_models, temp_non_skin_histogram
                )
        skin_models = np.divide(skin_models, sum(sum(skin_models)))
        non_skin_models = np.divide(non_skin_models, sum(sum(non_skin_models)))
