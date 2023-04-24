import cv2
import numpy as np
from matplotlib import pyplot as plt


# TODO ----------------------------------------------- Pochodne cząstkowe ----------------------------------------------

def zad_1():
    img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

    prewitt_x_mask = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]], np.int8)
    prewitt_y_mask = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]], np.int8)
    sobel_x_mask = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], np.int8)
    sobel_y_mask = np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], np.int8)

    prewitt_x = cv2.filter2D(img, cv2.CV_32F, prewitt_x_mask) / 3
    prewitt_y = cv2.filter2D(img, cv2.CV_32F, prewitt_y_mask) / 3
    sobel_x = cv2.filter2D(img, cv2.CV_32F, sobel_x_mask) / 4
    sobel_y = cv2.filter2D(img, cv2.CV_32F, sobel_y_mask) / 4

    cv2.imshow('prewitt x', abs(prewitt_x).astype(np.uint8))
    cv2.imshow('prewitt y', abs(prewitt_y).astype(np.uint8))
    cv2.imshow('sobel x', abs(sobel_x).astype(np.uint8))
    cv2.imshow('sobel y', abs(sobel_y).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ------------------------------------------- Moduł i kierunek gradientu ------------------------------------------

def zad_2():
    img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    prewitt_x_mask = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]], np.int8)
    prewitt_y_mask = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]], np.int8)
    prewitt_x = cv2.filter2D(img, cv2.CV_32F, prewitt_x_mask) / 3
    prewitt_y = cv2.filter2D(img, cv2.CV_32F, prewitt_y_mask) / 3

    gradient = cv2.sqrt(cv2.pow(prewitt_x, 2) + cv2.pow(prewitt_y, 2))
    print(gradient[100:105, 100:105])

    cv2.imshow('img_gradient', gradient.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ------------------------------------- Wykrywanie krawędzi - metoda Canny’ego ------------------------------------

def zad_3():
    def callback(value):
        pass

    img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('Canny')
    cv2.createTrackbar('Lower threshold', 'Canny', 0, 255, callback)
    cv2.createTrackbar('Upper threshold', 'Canny', 0, 255, callback)

    while True:
        t_lower = cv2.getTrackbarPos('Lower threshold', 'Canny')
        t_upper = cv2.getTrackbarPos('Upper threshold', 'Canny')
        canny = cv2.Canny(img, t_lower, t_upper)
        cv2.imshow('Canny', canny)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


# TODO --------------------- Wykrywanie linii prostych oraz okręgów przy pomocy transformaty Hough’a -------------------

def zad_4():
    shapes = cv2.imread('shapes.jpg')
    prob_shapes = shapes.copy()
    circ_shapes = shapes.copy()
    shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
    shapes_blur = cv2.medianBlur(shapes_gray, 5)
    edges = cv2.Canny(shapes_gray, 50, 150, apertureSize=3)
    hough_lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 200)
    prob_hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=10, maxLineGap=10)
    hough_circles = cv2.HoughCircles(shapes_blur, cv2.HOUGH_GRADIENT, 1, 70, param1=240, param2=50, minRadius=10,
                                     maxRadius=100)

    d = 2000
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + d * (-b))
        y1 = int(y0 + d * a)
        x2 = int(x0 - d * (-b))
        y2 = int(y0 - d * a)
        cv2.line(shapes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line in prob_hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(prob_shapes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hough_circles = np.uint16(np.around(hough_circles))
    for i in hough_circles[0, :]:
        # draw the outer circle
        cv2.circle(circ_shapes, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(circ_shapes, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Hough transform', shapes)
    cv2.imshow('Probabilistic Hough transform', prob_shapes)
    cv2.imshow('Hough circle transform', circ_shapes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO --------------------------------------- Zadania do samodzielnej realizacji --------------------------------------

def zad_5():
    ship = cv2.imread('drone_ship.jpg')
    ship_gray = cv2.cvtColor(ship, cv2.COLOR_BGR2GRAY)
    ship_blur = cv2.medianBlur(ship_gray, 5)
    edges = cv2.Canny(ship_blur, 70, 150)
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=40, maxLineGap=10)

    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(ship, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Drone ship', ship)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_6():
    fruit = cv2.imread('fruit.jpg')
    fruit_gray = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    fruit_blur = cv2.medianBlur(fruit_gray, 5)
    hough_circles = cv2.HoughCircles(fruit_blur, cv2.HOUGH_GRADIENT, 1, 70, param1=240, param2=50, minRadius=70,
                                     maxRadius=250)

    hough_circles = np.uint16(np.around(hough_circles))
    for i in hough_circles[0, :]:
        b, g, r = fruit[i[1] - 50, i[0]]
        if g > 190:
            # draw the outer circle
            cv2.circle(fruit, (i[0], i[1]), i[2], (0, 255, 0), 2)
        else:
            # draw the outer circle
            cv2.circle(fruit, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # draw the center of the circle
        cv2.circle(fruit, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Fruit', fruit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_7():
    coins = cv2.imread('coins.jpg')
    coins_gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    coins_blur = cv2.medianBlur(coins_gray, 5)
    hough_circles = cv2.HoughCircles(coins_blur, cv2.HOUGH_GRADIENT, 1, 70, param1=240, param2=50, minRadius=40,
                                     maxRadius=150)

    coins_sum = 0.0
    hough_circles = np.uint16(np.around(hough_circles))
    for i in hough_circles[0, :]:

        if 40 < i[2] < 60:
            coins_sum += 0.1
        elif 90 < i[2] < 110:
            coins_sum += 1.0

        # draw the outer circle
        cv2.circle(coins, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(coins, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f'Coins value: {round(coins_sum, 2)}')
    cv2.imshow('Fruit', coins)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_7()
