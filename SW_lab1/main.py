import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


# TODO ---------------------------------------------- Pierwszy program -------------------------------------------------

def zad_1():
    cap = cv2.VideoCapture(0)  # open the default camera

    key = ord('a')
    while key != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame comes here
        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        # Detect edges on the blurred image
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        # Display the result of our processing
        cv2.imshow('result', img_edges)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30)

    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


# TODO ------------------------------ Wczytywanie, zapisywanie i wyświetlanie obrazów ----------------------------------

def zad_2():
    img = cv2.imread('photo.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('photo2.jpg', img)


# TODO ----------------------------------------------- Obrazy a piksele ------------------------------------------------

def zad_3():
    img_color = cv2.imread('photo.jpg', cv2.IMREAD_COLOR)
    img_grayscale = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

    print(f'Color image parameters: {img_color.shape}')
    print(f'Grayscale image parameters: {img_grayscale.shape}')

    print(f'Pixel (220, 270) value - color: {img_color[220, 270]} , grayscale: {img_grayscale[220, 270]}')

    crop = img_color[119:253, 325:392]
    modified = img_color.copy()
    modified[35:169, 440:507] = crop
    cv2.imshow('crop', crop)
    cv2.imshow('modified', modified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('img_color', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(img_color)
    plt.show()

    img_bgr = cv2.imread('AdditiveColor.png', cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5)
    # b, g, r = cv2.split(img_bgr)
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]
    cv2.imshow('img_bgr', img_bgr)
    cv2.imshow('B', b)
    cv2.imshow('G', g)
    cv2.imshow('R', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ------------------------------------------- Obsługa kamer oraz wideo --------------------------------------------

def zad_4():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord(' ')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def zad_5():
    cap = cv2.VideoCapture('Wildlife.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# TODO ------------------------------------ Zadania do samodzielnej realizacji -----------------------------------------

def zad_6():
    number = 0
    image_list = []
    for file in os.listdir('.'):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
            image_list.append(cv2.imread(file))

    while True:
        cv2.imshow('image', image_list[number])
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('w'), 27]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            number = (number - 1) % len(image_list)
        elif key == ord('w'):
            number = (number + 1) % len(image_list)
        elif key == 27:
            break


if __name__ == '__main__':
    zad_6()
