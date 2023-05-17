import cv2
import numpy as np


# TODO ---------------------------------------- Zadanie 1 - znojdowanie obiektów ---------------------------------------

def zad_1():
    def cut_img(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, center=(x, y), radius=1, color=(0, 0, 255), thickness=6)
            pts.append((x, y))

    img = cv2.imread('matching.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
    img_rot = cv2.imread('matching_rotated.jpg', cv2.IMREAD_COLOR)
    img_rot = cv2.resize(img_rot, dsize=(0, 0), fx=0.3, fy=0.3)
    img_rot_gray = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
    img_rot_2 = img_rot.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', cut_img)
    pts = []

    while True:
        cv2.imshow('image', img)
        if len(pts) == 2:
            break
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()

    obj = img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
    obj_gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    obj_2 = obj.copy()

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(obj_gray, None)
    kp2, des2 = sift.detectAndCompute(img_rot_gray, None)
    kp_obj = cv2.drawKeypoints(obj, kp1, 0)
    kp_img_rot = cv2.drawKeypoints(img_rot, kp2, 0)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:20]
    matches_img = cv2.drawMatches(kp_obj, kp1, kp_img_rot, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w, ch = obj.shape
    src_box = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst_box = cv2.perspectiveTransform(src_box, H)
    orb_img_rot_2 = cv2.polylines(img_rot_2, [np.int32(dst_box)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    matches_img_2 = cv2.drawMatches(obj_2, kp1, orb_img_rot_2, kp2, good, None, **draw_params)

    cv2.imshow('matches', matches_img_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO ----------------------------------------- Zadanie 2 - tworzenie panoramy ----------------------------------------

def zad_2():
    img_1 = cv2.imread('panorama_1.jpg', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('panorama_2.jpg', cv2.IMREAD_COLOR)
    img_1 = cv2.resize(img_1, dsize=(0, 0), fx=0.5, fy=0.5)
    img_2 = cv2.resize(img_2, dsize=(0, 0), fx=0.5, fy=0.5)
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1_gray, None)
    kp2, des2 = sift.detectAndCompute(img_2_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:50]
    matches_img = cv2.drawMatches(img_1, kp1, img_2, kp2, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    cv2.imshow('matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    height = img_1.shape[0] + img_2.shape[0]
    width = img_1.shape[1] + img_2.shape[1]

    img_2_dst = cv2.warpPerspective(img_2, H, (width, height))
    img_2_dst[0:img_1.shape[0], 0:img_1.shape[1]] = img_1

    cv2.imshow('dst', img_2_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_2()
