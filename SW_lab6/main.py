import cv2
import numpy as np
from matplotlib import pyplot as plt


# TODO ----------------------------------- Wykrywanie konturów i analiza strukturalna ----------------------------------

def zad_1():
    img = cv2.imread('not_bad.jpg')
    img = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val, thresh_img = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(thresh_img, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(erosion, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
    ]

    img_with_contours = img.copy()
    src_points = []
    for cnt, color in zip(contours, colors):
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            cv2.circle(img_with_contours, (cx, cy), 2, color)
            src_points.append((cx, cy))
        cv2.drawContours(img_with_contours, [cnt], 0, color)

    cv2.imshow('contours', img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    src_points = np.float32(src_points)
    dst_points = np.float32([(700, 500), (0, 500), (700, 0), (0, 0)])
    perspective_transform = cv2.getPerspectiveTransform(src_points, dst_points)
    img_transformed = cv2.warpPerspective(img, perspective_transform, (700, 500))

    cv2.imshow('img_transformed', img_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ---------------------------------------------- Dopasowywanie wzorców --------------------------------------------

def zad_2():
    img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    template = img[60:121, 446:496]
    h, w = template.shape

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    cv2.imshow('res', res)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_3():
    img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
    template = img[60:121, 446:496]
    h, w = template.shape

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)

    cv2.imshow('res', res)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ----------------------------------------- OpenCV Tutorial Contour features --------------------------------------

def zad_4():
    # Moments
    shape04 = cv2.imread('shape04.jpg', cv2.IMREAD_GRAYSCALE)
    ret, shape04_thresh = cv2.threshold(shape04, 127, 255, 0)
    shape04_thresh_rgb = cv2.cvtColor(shape04_thresh, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(shape04_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    print('Shape 4 hierarchy: ', hierarchy)

    shape04_cnt = contours[0]
    shape04_M = cv2.moments(shape04_cnt)
    print('Shape 4 moments: ', shape04_M)

    # Contour Area
    area = cv2.contourArea(shape04_cnt)
    print('Shape 4 area: ', area)

    # Contour Perimeter
    perimeter = cv2.arcLength(shape04_cnt, True)
    print('Shape 4 perimeter: ', perimeter)

    # Contour Approximation
    shape01 = cv2.imread('shape01.jpg', cv2.IMREAD_GRAYSCALE)
    ret, shape01_thresh = cv2.threshold(shape01, 127, 255, 0)
    shape01_thresh_rgb = cv2.cvtColor(shape01_thresh, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(shape01_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    shape01_cnt = contours[0]

    epsilon = 0.01 * cv2.arcLength(shape01_cnt, True)
    approx = cv2.approxPolyDP(shape01_cnt, epsilon, True)
    cv2.drawContours(shape01_thresh_rgb, [shape01_cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(shape01_thresh_rgb, [approx], 0, (0, 0, 255), 2)

    cv2.imshow('contour approximation', shape01_thresh_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convex Hull
    hull = cv2.convexHull(shape04_cnt)
    cv2.drawContours(shape04_thresh_rgb, [hull], 0, (0, 0, 255), 2)

    cv2.imshow('convex hull', shape04_thresh_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Checking Convexity
    k = cv2.isContourConvex(shape04_cnt)
    print('Is shape 4 convex?: ', k)

    # Straight Bounding Rectangle
    shape02 = cv2.imread('shape02.jpg', cv2.IMREAD_GRAYSCALE)
    ret, shape02_thresh = cv2.threshold(shape02, 127, 255, 0)
    shape02_thresh_rgb = cv2.cvtColor(shape02_thresh, cv2.COLOR_GRAY2RGB)
    shape02_thresh_rgb_1 = shape02_thresh_rgb.copy()
    contours, hierarchy = cv2.findContours(shape02_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    shape02_cnt = contours[0]

    x, y, w, h = cv2.boundingRect(shape02_cnt)
    cv2.rectangle(shape02_thresh_rgb_1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('straight bounding rectangle', shape02_thresh_rgb_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Rotated Rectangle
    shape02_thresh_rgb_2 = shape02_thresh_rgb.copy()
    rect = cv2.minAreaRect(shape02_cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(shape02_thresh_rgb_2, [box], 0, (0, 0, 255), 2)

    cv2.imshow('rotated rectangle', shape02_thresh_rgb_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Minimum Enclosing Circle
    shape02_thresh_rgb_3 = shape02_thresh_rgb.copy()
    (x, y), radius = cv2.minEnclosingCircle(shape02_cnt)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(shape02_thresh_rgb_3, center, radius, (0, 255, 0), 2)

    cv2.imshow('minimum enclosing circle', shape02_thresh_rgb_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Fitting an Ellipse
    shape02_thresh_rgb_4 = shape02_thresh_rgb.copy()
    ellipse = cv2.fitEllipse(shape02_cnt)
    cv2.ellipse(shape02_thresh_rgb_4, ellipse, (0, 255, 0), 2)

    cv2.imshow('ellipse', shape02_thresh_rgb_4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Fitting a Line
    shape02_thresh_rgb_5 = shape02_thresh_rgb.copy()
    rows, cols = shape02_thresh_rgb_5.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(shape02_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(shape02_thresh_rgb_5, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    cv2.imshow('line', shape02_thresh_rgb_5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO --------------------------------------- OpenCV Tutorial Contour properties --------------------------------------

def zad_5():
    # Aspect ratio
    shape03 = cv2.imread('shape03.jpg', cv2.IMREAD_GRAYSCALE)
    ret, shape03_thresh = cv2.threshold(shape03, 127, 255, 0)
    shape03_thresh_rgb = cv2.cvtColor(shape03_thresh, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(shape03_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    shape03_cnt = contours[0]

    x, y, w, h = cv2.boundingRect(shape03_cnt)
    aspect_ratio = float(w) / h
    print('Aspect ratio: ', aspect_ratio)

    # Extent
    area = cv2.contourArea(shape03_cnt)
    rect_area = w * h
    extent = float(area) / rect_area
    print('Extent: ', extent)

    # Solidity
    hull = cv2.convexHull(shape03_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    print('Solidity: ', solidity)

    # Equivalent Diameter
    equi_diameter = np.sqrt(4 * area / np.pi)
    print('Equivalent diameter: ', equi_diameter)

    # Orientation
    (x, y), (MA, ma), angle = cv2.fitEllipse(shape03_cnt)
    print('Orientation: ', angle)

    # Mask and Pixel Points
    mask = np.zeros(shape03_thresh.shape, np.uint8)
    cv2.drawContours(mask, [shape03_cnt], 0, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask))
    print(pixelpoints)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Maximum Value, Minimum Value and their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(shape03, mask=mask)
    print(min_val)

    # Mean Color or Mean Intensity
    mean_val = cv2.mean(shape03, mask=mask)
    print(mean_val)

    # Extreme Points
    leftmost = tuple(shape03_cnt[shape03_cnt[:, :, 0].argmin()][0])
    rightmost = tuple(shape03_cnt[shape03_cnt[:, :, 0].argmax()][0])
    topmost = tuple(shape03_cnt[shape03_cnt[:, :, 1].argmin()][0])
    bottommost = tuple(shape03_cnt[shape03_cnt[:, :, 1].argmax()][0])

    cv2.circle(shape03_thresh_rgb, leftmost, 6, (0, 0, 255), cv2.FILLED)
    cv2.circle(shape03_thresh_rgb, rightmost, 6, (0, 0, 255), cv2.FILLED)
    cv2.circle(shape03_thresh_rgb, topmost, 6, (0, 0, 255), cv2.FILLED)
    cv2.circle(shape03_thresh_rgb, bottommost, 6, (0, 0, 255), cv2.FILLED)

    cv2.imshow('img', shape03_thresh_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ------------------------------------ OpenCV Tutorial Contours more properties  ----------------------------------

def zad_6():
    # Convexity Defects
    shape04 = cv2.imread('shape04.jpg', cv2.IMREAD_GRAYSCALE)
    ret, shape04_thresh = cv2.threshold(shape04, 127, 255, 0)
    shape04_thresh_rgb = cv2.cvtColor(shape04_thresh, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(shape04_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    shape04_cnt = contours[0]

    hull = cv2.convexHull(shape04_cnt, returnPoints=False)
    defects = cv2.convexityDefects(shape04_cnt, hull)

    print('Defects: ', defects)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(shape04_cnt[s][0])
        end = tuple(shape04_cnt[e][0])
        far = tuple(shape04_cnt[f][0])
        cv2.line(shape04_thresh_rgb, start, end, [0, 255, 0], 2)
        cv2.circle(shape04_thresh_rgb, far, 5, [0, 0, 255], -1)

    cv2.imshow('img', shape04_thresh_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Point Polygon Test
    dist = cv2.pointPolygonTest(shape04_cnt, (50, 50), True)
    print('Distance: ', dist)


if __name__ == '__main__':
    zad_6()

