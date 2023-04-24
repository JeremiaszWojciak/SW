import cv2
import numpy as np
import time

# TODO ----------------------------------- Obsługa zdarzeń interfejsu - trackbar ---------------------------------------


def zad_1():
    def r_callback(value):
        print('R: ', value)

    def g_callback(value):
        print('G: ', value)

    def b_callback(value):
        print('B: ', value)

    def s_callback(value):
        print('Switch: ', value)

    # create a black image, a window
    img = np.zeros((300, 512, 3), dtype=np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, r_callback)
    cv2.createTrackbar('G', 'image', 0, 255, g_callback)
    cv2.createTrackbar('B', 'image', 0, 255, b_callback)

    # create switch for ON/OFF functionality
    switch_trackbar_name = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch_trackbar_name, 'image', 0, 1, s_callback)

    while True:
        cv2.imshow('image', img)

        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch_trackbar_name, 'image')

        if s == 0:
            # assign zeros to all pixels
            img[:] = 0
        else:
            # assign the same BGR color to all pixels
            img[:] = [b, g, r]

    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()


# TODO ------------------------------------- Operacje na pikselach - progowanie ----------------------------------------

def zad_2():
    def thresh_callback(value):
        pass

    cv2.namedWindow('threshold image')
    cv2.createTrackbar('Threshold', 'threshold image', 0, 255, thresh_callback)
    cv2.createTrackbar('Mode', 'threshold image', 0, 4, thresh_callback)

    img = cv2.imread('photo.jpg', cv2.COLOR_BGR2GRAY)
    thresh_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO,
                    cv2.THRESH_TOZERO_INV]

    while True:
        thresh_val = cv2.getTrackbarPos('Threshold', 'threshold image')
        thresh_type = cv2.getTrackbarPos('Mode', 'threshold image')

        thresh_val, thresh_img = cv2.threshold(img, thresh_val, 255, thresh_types[thresh_type])
        cv2.imshow('threshold image', thresh_img)
        key_code = cv2.waitKey(10)
        if key_code == 27:
            break
    cv2.destroyAllWindows()


# TODO ------------------------------------ Zmiana rozmiaru - skalowanie obrazów ---------------------------------------

def zad_3():
    img = cv2.imread('cube.jpg')
    fx = 2.75
    fy = 2.75

    scaled_linear = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    scaled_nearest = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    scaled_area = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    scaled_lanczos4 = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('Bilinear', scaled_linear)
    cv2.imshow('Nearest neighbor', scaled_nearest)
    cv2.imshow('Area', scaled_area)
    cv2.imshow('Lanczos', scaled_lanczos4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ------------------------------------------- Przykład dodawania obrazów ------------------------------------------

def zad_4():
    img = cv2.imread('cube.jpg', cv2.IMREAD_GRAYSCALE)

    img_add_cv = cv2.add(img, 40)
    img_add_float = img.astype(np.float32) + 40
    img_add = img + 40

    cv2.imshow('img', img)
    cv2.imshow('img_add_cv', img_add_cv)
    cv2.imshow('img_add', img_add)
    cv2.imshow('img_add_float', img_add_float / (255 + 40))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO -------------------------------- Obrazy jako macierze - operacje matematyczne -----------------------------------

def zad_5():
    def callback(value):
        pass

    cv2.namedWindow('image')
    cv2.createTrackbar('Alpha', 'image', 0, 100, callback)
    cube = cv2.imread('cube.jpg')
    logo = cv2.imread('PUTVISION_LOGO.png')
    logo = cv2.resize(logo, dsize=(cube.shape[1], cube.shape[0]))

    while True:
        alpha = cv2.getTrackbarPos('Alpha', 'image') / 100
        blended = cv2.addWeighted(cube, alpha, logo, 1-alpha, 0)
        cv2.imshow('image', blended)

        key_code = cv2.waitKey(10)
        if key_code == 27:
            break
    cv2.destroyAllWindows()


# TODO ----------------------- Zadania do samodzielnej realizacji - pomiar czasu interpolacji --------------------------

def zad_6():
    img = cv2.imread('cube.jpg')
    fx = 2.75
    fy = 2.75

    linear_start = time.time()
    cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    linear_stop = time.time()

    nearest_start = time.time()
    cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    nearest_stop = time.time()

    area_start = time.time()
    cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    area_stop = time.time()

    lanczos4_start = time.time()
    cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)
    lanczos4_stop = time.time()

    print('Bilinear interpolation time: ', round((linear_stop - linear_start) * 1000, 3), 'ms')
    print('Nearest neighbor interpolation time: ', round((nearest_stop - nearest_start) * 1000, 3), 'ms')
    print('Area interpolation time: ', round((area_stop - area_start) * 1000, 3), 'ms')
    print('Lanczos4 interpolation time: ', round((lanczos4_stop - lanczos4_start) * 1000, 3), 'ms')


# TODO ---------------------------- Zadania do samodzielnej realizacji - negatyw obrazu --------------------------------

def zad_7():
    img_color = cv2.imread('photo.jpg', cv2.IMREAD_COLOR)
    # negative = cv2.subtract(full, img_color)
    negative_rgb = 255 - img_color
    negative_grayscale = 255 - cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow('negative rgb', negative_rgb)
    cv2.imshow('negative grayscale', negative_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_7()


