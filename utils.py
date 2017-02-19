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


def inverse_image(I):
    # I : cv2 img
    return 255 - I


def BGR_to_HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def BGR_to_Lab(img):
    return cv2.cvtColor(img, cv2.CV_BGR2Lab)


# http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
def BGR_histogram(bgr_img, black_and_white_img):
    # plot a 2D color histogram for red and green
    hist = cv2.calcHist([bgr_img], [2, 1], black_and_white_img[:, :, 0],
                        [32, 32], [0, 256, 0, 256])
    return hist


def HSV_histogram(hsv_img, black_and_white_img):
    hist = cv2.calcHist([hsv_img], [0, 1], black_and_white_img[:, :, 0],
                        [180, 256], [0, 180, 0, 256])
    return hist

if __name__ == '__main__':
    # data/Ground_Truth/GroundT_FacePhoto/06Apr03Face.png
    # data/Pratheepan_Dataset/FacePhoto/06Apr03Face.jpg
    bgr_img = load_image("data/Pratheepan_Dataset/FacePhoto/06Apr03Face.jpg",
                         True)
    hsv_img = BGR_to_HSV(bgr_img)
    black_and_white_img = load_image(
        "data/Ground_Truth/GroundT_FacePhoto/06Apr03Face.png", True)
    '''cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    histo = HSV_histogram(hsv_img, black_and_white_img)
    print(histo)
    print(histo.shape)
