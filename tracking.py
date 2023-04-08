import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import loadData
from progress.bar import Bar


def calculate_track(images, detector_name, projection_mat, intrinsic_mat, initial_pose, threshold=0.5):
    current_pose = initial_pose
    track = [current_pose]
    detector, matcher = create_detector_and_matcher(detector_name)
    counter_matches = 0
    bar = Bar('Обработка изображений:', max=len(images))
    for i, image in enumerate(images):
        bar.next()
        if i == 0:
            continue
        else:
            pts1, pts2 = find_matches(images[i-1], images[i], detector, matcher, threshold)
            counter_matches += len(pts1)
            transform_mat = get_transform_mat(pts1, pts2, intrinsic_mat)
            current_pose = np.matmul(current_pose, np.linalg.inv(transform_mat))
            track.append(current_pose)
    bar.finish()
    print('Среднее количество "хороших" совпадений: ', counter_matches/len(images))
    return track


def create_detector_and_matcher(detector_name):
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    if detector_name == 'sift':
        sift = cv.SIFT_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return sift, flann
    elif detector_name == 'surf':
        surf = cv.xfeatures2d.SURF_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=300)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        # bf = cv.BFMatcher()
        return surf, flann
    elif detector_name == 'orb':
        orb = cv.ORB_create()
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=300)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        # bf = cv.BFMatcher()
        return orb, flann
    elif detector_name == 'brisk':
        brisk = cv.BRISK_create()
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return brisk, flann
    elif detector_name == 'kaze':
        kaze = cv.KAZE_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return kaze, flann


def find_matches(img1, img2, detector, matcher, threshold=1):
    keypoint1, descriptor1 = detector.detectAndCompute(img1, None)
    keypoint2, descriptor2 = detector.detectAndCompute(img2, None)
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    # -- Filter matches using the Lowe's ratio test
    pts1 = []
    pts2 = []
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < threshold*n.distance:
                pts2.append(keypoint2[m.trainIdx].pt)
                pts1.append(keypoint1[m.queryIdx].pt)
        except ValueError:
            pass
    return pts1, pts2


def get_transform_mat(pts1, pts2, intrinsic_mat):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat, 1, 0.95)

    R = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat, np.array(0))

    t = R[2]
    R = R[1]

    return form_transf(R, np.ndarray.flatten(t))


def test_find_matches():
    img1 = cv.imread('./sample/dataset/00/image_0/000000.png')
    img2 = cv.imread('./sample/dataset/00/image_0/000001.png')

    pts1, pts2 = find_matches(img1, img2)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    projection_mat, intrinsic_mat = loadData.load_calib('./sample/dataset/00/calib.txt')

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat)

    R = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat, np.array(0))

    t = R[2]
    R = R[1]

    return form_transf(R, np.ndarray.flatten(t))


def form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


if __name__ == '__main__':
    sys.exit(test_find_matches())


def draw_matches(img1, img2, threshold=0.8):
    sift = cv.SIFT_create()
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    # -- Filter matches using the Lowe's ratio test
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor = (0, 255, 0),
                       singlePointColor = (255, 0, 0),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_DEFAULT)

    img3 = cv.drawMatchesKnn(img1, keypoint1, img2, keypoint2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()
