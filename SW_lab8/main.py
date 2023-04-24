import cv2
import time


# TODO -------------------------------------- Zadania do samodzielnej realizacji 1 -------------------------------------

def zad_1():
    img = cv2.imread('images/forward-1.bmp', cv2.IMREAD_COLOR)

    # Detekcja i wyświetlenie cech

    fast = cv2.FastFeatureDetector_create()
    fast_det_start = time.time()
    fast_kpts = fast.detect(img)
    fast_det_stop = time.time()
    fast_img = cv2.drawKeypoints(img, fast_kpts, 0)
    cv2.imshow('fast keypoints', fast_img)

    orb = cv2.ORB_create()
    orb_det_start = time.time()
    orb_kpts = orb.detect(img)
    orb_det_stop = time.time()
    orb_img = cv2.drawKeypoints(img, orb_kpts, 0)
    cv2.imshow('orb keypoints', orb_img)

    sift = cv2.SIFT_create()
    sift_det_start = time.time()
    sift_kpts = sift.detect(img)
    sift_det_stop = time.time()
    sift_img = cv2.drawKeypoints(img, sift_kpts, 0)
    cv2.imshow('sift keypoints', sift_img)

    # Porównanie liczby znalezionych cech

    print('Fast keypoints count: ', len(fast_kpts))
    print('Orb keypoints count: ', len(orb_kpts))
    print('Sift keypoints count: ', len(sift_kpts))

    # Porównanie czasu działania algorytmów detekcji

    print('Fast detection time: ', round((fast_det_stop - fast_det_start) * 100, 3), 'ms')
    print('Orb detection time: ', round((orb_det_stop - orb_det_start) * 100, 3), 'ms')
    print('Sift detection time: ', round((sift_det_stop - sift_det_start) * 100, 3), 'ms')

    # print(fast_kpts[0].angle)
    # print(fast_kpts[0].class_id)
    # print(fast_kpts[0].octave)
    # print(fast_kpts[0].pt)
    # print(fast_kpts[0].response)
    # print(fast_kpts[0].size)

    # Deskrypcja cech i wyświetlenie

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32)
    brief_des_start = time.time()
    fast_kpts, brief_des = brief.compute(img, fast_kpts)
    brief_des_stop = time.time()
    print('Brief descriptor values:\n ', brief_des)
    print('Brief descriptor shape:\n ', brief_des.shape)

    orb_des_start = time.time()
    orb_kpts, orb_des = orb.compute(img, orb_kpts)
    orb_des_stop = time.time()
    print('Orb descriptor values:\n ', orb_des)
    print('Orb descriptor shape:\n ', orb_des.shape)

    sift_des_start = time.time()
    sift_kpts, sift_des = sift.compute(img, sift_kpts)
    sift_des_stop = time.time()
    print('Sift descriptor values:\n ', sift_des)
    print('Sift descriptor shape:\n ', sift_des.shape)

    # Porównanie czasu działania algorytmów deskrypcji w przeliczeniu na 1 cechę

    print('Brief description time: ', round(((brief_des_stop - brief_des_start) * 100) / brief_des.shape[0], 4), 'ms')
    print('Orb description time: ', round(((orb_des_stop - orb_des_start) * 100) / orb_des.shape[0], 4), 'ms')
    print('Sift description time: ', round(((sift_des_stop - sift_des_start) * 100) / sift_des.shape[0], 4), 'ms')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_2():
    # Detektor i deskryptor ORB

    # img_1 = cv2.imread('images/forward-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/forward-2.bmp', cv2.IMREAD_COLOR)

    # img_1 = cv2.imread('images/perspective-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/perspective-2.bmp', cv2.IMREAD_COLOR)

    img_1 = cv2.imread('images/rotate-1.bmp', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('images/rotate-2.bmp', cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()
    orb_kpts_1, orb_des_1 = orb.detectAndCompute(img_1, None)
    orb_kpts_2, orb_des_2 = orb.detectAndCompute(img_2, None)

    # Dopasowanie cech

    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.match(orb_des_1, orb_des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches_img = cv2.drawMatches(img_1, orb_kpts_1, img_2, orb_kpts_2, matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # print(matches[0].distance)
    # print(matches[0].imgIdx)
    # print(matches[0].queryIdx)
    # print(matches[0].trainIdx)

    cv2.imshow('matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_3():
    # Detektor i deskryptor SIFT

    # img_1 = cv2.imread('images/forward-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/forward-2.bmp', cv2.IMREAD_COLOR)

    # img_1 = cv2.imread('images/perspective-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/perspective-2.bmp', cv2.IMREAD_COLOR)

    img_1 = cv2.imread('images/rotate-1.bmp', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('images/rotate-2.bmp', cv2.IMREAD_COLOR)

    sift = cv2.SIFT_create()
    sift_kpts_1, sift_des_1 = sift.detectAndCompute(img_1, None)
    sift_kpts_2, sift_des_2 = sift.detectAndCompute(img_2, None)

    # Dopasowanie cech

    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.match(sift_des_1, sift_des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches_img = cv2.drawMatches(img_1, sift_kpts_1, img_2, sift_kpts_2, matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def zad_4():
    # Detektor FAST i deskryptor BRIEF

    # img_1 = cv2.imread('images/forward-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/forward-2.bmp', cv2.IMREAD_COLOR)

    # img_1 = cv2.imread('images/perspective-1.bmp', cv2.IMREAD_COLOR)
    # img_2 = cv2.imread('images/perspective-2.bmp', cv2.IMREAD_COLOR)

    img_1 = cv2.imread('images/rotate-1.bmp', cv2.IMREAD_COLOR)
    img_2 = cv2.imread('images/rotate-2.bmp', cv2.IMREAD_COLOR)

    fast = cv2.FastFeatureDetector_create()
    fast_kpts_1 = fast.detect(img_1)
    fast_kpts_2 = fast.detect(img_2)

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32)
    fast_kpts_1, brief_des_1 = brief.compute(img_1, fast_kpts_1)
    fast_kpts_2, brief_des_2 = brief.compute(img_2, fast_kpts_2)

    # Dopasowanie cech

    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.match(brief_des_1, brief_des_2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches_img = cv2.drawMatches(img_1, fast_kpts_1, img_2, fast_kpts_2, matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('matches', matches_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    zad_4()
