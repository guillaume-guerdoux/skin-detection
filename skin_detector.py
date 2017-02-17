import numpy as np
from matplotlib import pyplot as plt
import cv2


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
