import cv2
import numpy as np
from matplotlib import pyplot as plt


# TODO ------------------------------------ Zadania do samodzielnej realizacji 1, 2 ------------------------------------
# Odejmowanie kolejnych klatek

def zad_1():
    def callback(value):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background image')
    cv2.namedWindow('current image')
    cv2.namedWindow('foreground image')
    cv2.createTrackbar('Threshold', 'foreground image', 0, 255, callback)
    cv2.setTrackbarPos('Threshold', 'foreground image', 30)
    background_img = None
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, current_img = cap.read()
        if not ret:
            break
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        x = cv2.getTrackbarPos('Threshold', 'foreground image')

        if background_img is not None:
            foreground_img = cv2.absdiff(background_img, current_img)
            foreground_img = cv2.dilate(foreground_img, kernel, iterations=1)
            foreground_img = cv2.erode(foreground_img, kernel, iterations=1)
            ret, foreground_img = cv2.threshold(foreground_img, x, 255, 0)

            cv2.imshow('background image', background_img)
            cv2.imshow('current image', current_img)
            cv2.imshow('foreground image', foreground_img)

        background_img = current_img

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# TODO -------------------------------------- Zadania do samodzielnej realizacji 3 -------------------------------------
# Algorytm przybliżonej mediany

def zad_2():
    def callback(value):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background image')
    cv2.namedWindow('current image')
    cv2.namedWindow('foreground image')
    cv2.createTrackbar('Threshold', 'foreground image', 0, 255, callback)
    cv2.setTrackbarPos('Threshold', 'foreground image', 30)
    background_img = None
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, current_img = cap.read()
        if not ret:
            break
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        x = cv2.getTrackbarPos('Threshold', 'foreground image')

        if background_img is not None:
            foreground_img = cv2.absdiff(background_img, current_img)
            foreground_img = cv2.dilate(foreground_img, kernel, iterations=1)
            foreground_img = cv2.erode(foreground_img, kernel, iterations=1)
            ret, foreground_img = cv2.threshold(foreground_img, x, 255, 0)

            background_img = np.where(background_img < current_img, background_img + 1, background_img)
            background_img = np.where(background_img > current_img, background_img - 1, background_img)

            cv2.imshow('background image', background_img)
            cv2.imshow('current image', current_img)
            cv2.imshow('foreground image', foreground_img)
        else:
            background_img = current_img

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# TODO -------------------------------------- Zadania do samodzielnej realizacji 4 -------------------------------------
# Algorytm mieszanin gaussowskich

def zad_3():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('current image')
    cv2.namedWindow('foreground image')
    backSub = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, current_img = cap.read()
        if not ret:
            break
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        foreground_img = backSub.apply(current_img)

        cv2.imshow('current image', current_img)
        cv2.imshow('foreground image', foreground_img)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# TODO -------------------------------------- Zadania do samodzielnej realizacji 5 -------------------------------------
# Alarm i śledzenie

def zad_4():
    def callback(value):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background image')
    cv2.namedWindow('current image')
    cv2.namedWindow('foreground image')
    cv2.createTrackbar('Threshold', 'foreground image', 0, 255, callback)
    cv2.setTrackbarPos('Threshold', 'foreground image', 30)
    background_img = None

    while cap.isOpened():
        ret, current_img_rgb = cap.read()
        if not ret:
            break
        current_img = cv2.cvtColor(current_img_rgb, cv2.COLOR_BGR2GRAY)
        x = cv2.getTrackbarPos('Threshold', 'foreground image')

        if background_img is not None:
            foreground_img = cv2.absdiff(background_img, current_img)
            ret, foreground_img = cv2.threshold(foreground_img, x, 255, 0)

            pixel_count = np.count_nonzero(foreground_img == 255)
            if pixel_count > 5000:
                print('ALARM!!!')
                indices = np.argwhere(foreground_img == 255)
                max_id = np.amax(indices, axis=0)
                min_id = np.amin(indices, axis=0)
                cv2.rectangle(current_img_rgb, (min_id[1], min_id[0]), (max_id[1], max_id[0]), (0, 255, 0), 2)

            background_img = np.where(background_img < current_img, background_img + 1, background_img)
            background_img = np.where(background_img > current_img, background_img - 1, background_img)

            cv2.imshow('background image', background_img)
            cv2.imshow('current image', current_img_rgb)
            cv2.imshow('foreground image', foreground_img)
        else:
            background_img = current_img

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# TODO ------------------------------------------------ Zadania domowe 1 -----------------------------------------------
# Zmodyfikowany obraz różnicowy

def zad_5():
    def callback(value):
        pass

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('foreground image')
    cv2.createTrackbar('Threshold', 'foreground image', 0, 255, callback)
    cv2.setTrackbarPos('Threshold', 'foreground image', 30)
    previous_img = None
    current_img = None
    kernel = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, next_img = cap.read()
        if not ret:
            break
        next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        x = cv2.getTrackbarPos('Threshold', 'foreground image')

        if current_img is not None and previous_img is not None:
            diff_1 = cv2.absdiff(next_img, current_img)
            diff_2 = cv2.absdiff(next_img, previous_img)
            foreground_img = cv2.bitwise_and(diff_1, diff_2)

            foreground_img = cv2.dilate(foreground_img, kernel, iterations=1)
            foreground_img = cv2.erode(foreground_img, kernel, iterations=1)
            ret, foreground_img = cv2.threshold(foreground_img, x, 255, 0)

            cv2.imshow('foreground image', foreground_img)

        previous_img = current_img
        current_img = next_img

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# TODO ------------------------------------------------ Zadania domowe 2 -----------------------------------------------
# Śledzenie kolorowych obiektów

def zad_6():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    lower = np.array([40, 100, 20], dtype=np.uint8)
    upper = np.array([80, 255, 255], dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        if np.count_nonzero(mask == 255) > 0:
            indices = np.argwhere(mask == 255)
            max_id = np.amax(indices, axis=0)
            min_id = np.amin(indices, axis=0)
            cv2.rectangle(output, (min_id[1], min_id[0]), (max_id[1], max_id[0]), (0, 255, 0), 2)

        cv2.imshow('image', output)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_6()
