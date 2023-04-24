import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import time
from scipy.signal import convolve2d


# TODO --------------------------------------------- Filtracja obrazów -------------------------------------------------

def zad_1():
    def callback():
        pass

    lenna_noise = cv2.imread('lenna_noise.bmp')
    lenna_sap = cv2.imread('lenna_salt_and_pepper.bmp')
    img = lenna_sap

    cv2.namedWindow('img')
    cv2.createTrackbar('kernel_size', 'img', 0, 10, callback)

    while True:
        kernel_size = 1 + 2 * cv2.getTrackbarPos('kernel_size', 'img')  # 1, 3, 5, 7, 9, 11

        img_after_blur = cv2.blur(img, (kernel_size, kernel_size))
        img_after_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        img_after_median = cv2.medianBlur(img, kernel_size)

        cv2.imshow('img', img)
        cv2.imshow('img_after_blur', img_after_blur)
        cv2.imshow('img_after_gaussian', img_after_gaussian)
        cv2.imshow('img_after_median', img_after_median)
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break
    cv2.destroyAllWindows()


# TODO ------------------------------------------- Operacje morfologiczne ----------------------------------------------

def zad_2():
    def callback(value):
        pass

    cv2.namedWindow('morphological operations')
    cv2.createTrackbar('Kernel size', 'morphological operations', 0, 10, callback)
    img = cv2.imread('morph.png')
    img = cv2.resize(img, dsize=(0, 0), fx=2, fy=2)

    while True:
        kernel_size = 1 + 2 * cv2.getTrackbarPos('Kernel size', 'morphological operations')
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # erosion = cv2.erode(img, kernel, iterations=1)
        # cv2.imshow('Morphological operations', erosion)

        # dilation = cv2.dilate(img, kernel, iterations=1)
        # cv2.imshow('Morphological operations', dilation)

        # erosion = cv2.erode(img, kernel, iterations=1)
        # opening = cv2.dilate(erosion, kernel, iterations=1)
        # cv2.imshow('Morphological operations', opening)

        dilation = cv2.dilate(img, kernel, iterations=1)
        closing = cv2.erode(dilation, kernel, iterations=1)
        cv2.imshow('morphological operations', closing)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break

    cv2.destroyAllWindows()


# TODO ---------------------------------------------- Skanowanie obrazu ------------------------------------------------

def zad_3():
    def average_filter(data):
        rows_, cols_ = data.shape
        new_img = np.zeros((rows_, cols_))
        for row in range(rows_):
            for col in range(cols_):
                if row == 0 or row == rows_ - 1 or col == 0 or col == cols_ - 1:
                    new_img[row][col] = data[row][col]
                else:
                    temp = (int(data[row - 1][col - 1]) + int(data[row - 1][col]) + int(data[row - 1][col + 1]) +
                            int(data[row][col - 1]) + int(data[row][col]) + int(data[row][col + 1]) +
                            int(data[row + 1][col - 1]) + int(data[row + 1][col]) + int(data[row + 1][col + 1])) / 9
                    new_img[row][col] = round(temp)
        new_img = new_img.astype(np.uint8)
        return new_img

    # Zamiana co trzeciego piksela w wierszu na biały

    gray = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)

    rows, cols = gray.shape
    for r in range(rows):
        for c in range(cols):
            if c % 3 == 0:
                gray[r][c] = 255

    # Filtracja własnym filtrem

    my_filter_start = time.time()
    my_filter_img = average_filter(gray)
    my_filter_stop = time.time()

    # Filtracja funkcją cv2.blur

    blur_start = time.time()
    blur_img = cv2.blur(gray, (3, 3))
    blur_stop = time.time()

    # Filtracja funkcją cv2.filter2D

    filter2d_start = time.time()
    kernel = np.full((3, 3), 1/9, dtype=np.float32)
    filter2d = cv2.filter2D(gray, -1, kernel)
    filter2d_stop = time.time()

    cv2.imshow('gray', gray)
    cv2.imshow('my filter', my_filter_img)
    cv2.imshow('blur', blur_img)
    cv2.imshow('filter2D', filter2d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('My filter time: ', round((my_filter_stop - my_filter_start) * 100, 3), 'ms')
    print('Blur time: ', round((blur_stop - blur_start) * 100, 3), 'ms')
    print('Filter2D time: ', round((filter2d_stop - filter2d_start) * 100, 3), 'ms')


# TODO -------------------------------------- Zadania do samodzielnej realizacji ---------------------------------------

def zad_4():
    def kuwahara(data, size):
        image = data.astype(np.float32)

        tmpAvgKerRow = np.hstack((np.ones((1, int((size-1)/2+1))), np.zeros((1, int((size-1)/2)))))
        tmpPadder = np.zeros((1, size))
        tmpavgker = np.tile(tmpAvgKerRow, (int((size - 1) / 2 + 1), 1))
        tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, (int((size - 1) / 2), 1))))
        tmpavgker = tmpavgker / np.sum(tmpavgker)
        print(tmpavgker)

        avgker = np.empty((4, size, size))
        avgker[0] = tmpavgker
        avgker[1] = np.fliplr(tmpavgker)
        avgker[2] = np.flipud(tmpavgker)
        avgker[3] = np.fliplr(avgker[2])

        squared_img = image ** 2

        avgs = np.zeros([4, image.shape[0], image.shape[1]])
        stddevs = avgs.copy()

        for k in range(4):
            avgs[k] = convolve2d(image, avgker[k], mode='same')             # mean on subwindow
            stddevs[k] = convolve2d(squared_img, avgker[k], mode='same')     # mean of squares on subwindow
            stddevs[k] = stddevs[k] - avgs[k] ** 2                          # variance on subwindow

        filtered = np.zeros(image.shape)
        indices = np.argmin(stddevs, axis=0)

        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                filtered[row, col] = avgs[indices[row, col], row, col]
        return filtered.astype(np.uint8)

    gray = cv2.imread('gray.png', cv2.IMREAD_GRAYSCALE)
    result = kuwahara(gray, 5)
    cv2.imshow('Kuwahara', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_4()
