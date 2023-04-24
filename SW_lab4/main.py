import cv2
import numpy as np
from matplotlib import pyplot as plt


# TODO ----------------------------------- Obsługa myszy i rysowanie figur na obrazie ----------------------------------

def zad_1():
    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(img, pt1=(x-10, y-10), pt2=(x+10, y+10), color=(255, 0, 0), thickness=1)
        elif event == cv2.EVENT_MBUTTONDOWN:
            cv2.circle(img, center=(x, y), radius=10, color=(0, 255, 0), thickness=1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


# TODO ------------------------------------------- Transformacje geometryczne ------------------------------------------

def zad_2():
    # upper left -> upper right -> lower left -> lower right

    def save_point(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(road, center=(x, y), radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
            src_pts.append((x, y))

    road = cv2.imread('road.jpg')
    road = cv2.resize(road, dsize=None, fx=0.5, fy=0.5)
    road_original = road.copy()
    src_pts = []
    road_dst = None

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', save_point)

    while True:
        cv2.imshow('image', road)
        if len(src_pts) == 4:
            src_pts = np.asarray(src_pts, dtype=np.float32)
            dst_pts = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            road_dst = cv2.warpPerspective(road_original, M, (300, 300))
            break
        if cv2.waitKey(10) == 27:
            break

    cv2.imshow('image_dst', road_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO --------------------------------------------------- Histogramy --------------------------------------------------

def zad_3():
    img_gray = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread('photo.jpg', cv2.IMREAD_COLOR)

    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    plt.subplot(121)
    plt.plot(hist_gray, color='k')
    plt.xlim([0, 256])

    color = ('b', 'g', 'r')
    plt.subplot(122)
    for i, col in enumerate(color):
        hist_i = cv2.calcHist([img_color], [i], None, [256], [0, 256])
        plt.plot(hist_i, color=col)
        plt.xlim([0, 256])
    plt.show()


def zad_4():
    img_gray = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(img_gray)
    res = np.vstack((img_gray, equalized_image))
    cv2.imshow('clahe equalization', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO --------------------------------------- Zadania do samodzielnej realizacji --------------------------------------

def zad_5():
    def save_point(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, center=(x, y), radius=1, color=(0, 0, 255), thickness=6)
            pts.append((x, y))

    img = cv2.imread('photo.jpg', cv2.IMREAD_COLOR)
    img_original = img.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', save_point)
    pts = []

    while True:
        cv2.imshow('image', img)
        if len(pts) == 2:
            roi = img_original[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
            (b, g, r) = cv2.split(roi)
            thresh_val, thresh_g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
            thresh_roi = cv2.merge((b, thresh_g, r))
            img_original[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]] = thresh_roi
            break
        if cv2.waitKey(10) == 27:
            break

    cv2.imshow('image', img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_6():
    def save_point(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(gallery, center=(x, y), radius=5, color=(0, 255, 0), thickness=cv2.FILLED)
            dst_pts.append((x, y))

    gallery = cv2.imread('gallery.png')
    gallery_rows, gallery_cols, gallery_ch = gallery.shape
    gallery_original = gallery.copy()
    pug = cv2.imread('pug.png')
    pug_rows, pug_cols, pug_ch = pug.shape
    dst_pts = []
    src_pts = np.float32([[0, 0], [pug_cols-1, 0], [0, pug_rows-1], [pug_cols-1, pug_rows-1]])
    pug_dst = None

    cv2.namedWindow('gallery')
    cv2.setMouseCallback('gallery', save_point)

    while True:
        cv2.imshow('gallery', gallery)
        if len(dst_pts) == 4:
            dst_pts = np.asarray(dst_pts, dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            pug_dst = cv2.warpPerspective(pug, M, (gallery_cols, gallery_rows))
            break
        if cv2.waitKey(10) == 27:
            break

    pug_gray = cv2.cvtColor(pug_dst, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(pug_gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    gallery_bg = cv2.bitwise_and(gallery_original, gallery_original, mask=mask_inv)
    gallery_pug = cv2.add(gallery_bg, pug_dst)
    cv2.imshow('gallery with pug', gallery_pug)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_6()
