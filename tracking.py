import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import loadData
from progress.bar import Bar


def calculate_track(images, detector_name, intrinsic_mat, initial_pose, ground_truth, threshold=1, inverse_mat=False):
    current_pose = initial_pose
    track = [current_pose]
    detector, matcher = create_detector_and_matcher(detector_name)
    counter_matches = 0
    bar = Bar('Обработка изображений детектором {}:'.format(detector_name), max=len(images))
    for i, image in enumerate(images):
        bar.next()
        if i == 0:
            continue
        else:
            pts1, pts2 = find_matches(images[i-1], images[i], detector, matcher, threshold)
            #pts1, pts2 = optical_flow(detector, images[i-1], images[i])
            counter_matches += len(pts1)
            R, t = get_R_T(pts1, pts2, intrinsic_mat)
            absolute_scale = get_absolute_scale(ground_truth[i - 1], ground_truth[i])
            t = absolute_scale * t
            transform_mat = form_transf(R, np.ndarray.flatten(t))
            if inverse_mat:
                current_pose = np.matmul(current_pose, np.linalg.inv(transform_mat))
            else:
                current_pose = np.matmul(current_pose, transform_mat)
            track.append(current_pose)
    bar.finish()
    print('Среднее количество "хороших" совпадений: ', counter_matches/len(images))
    return track


def track_points(detector, img1, img2, intrinsic_mat):
    points = detector.detect(img1)
    points = np.array([x.pt for x in points], dtype=np.float32).reshape(-1, 1, 2)

    lk_params = dict(winSize=(21, 21), criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    p1, st, err = cv.calcOpticalFlowPyrLK(img1, img2, points, None, **lk_params)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = points[st == 1]

    essential_mat, _ = cv.findEssentialMat(good_new, good_old, intrinsic_mat, cv.RANSAC)
    M = cv.recoverPose(essential_mat, good_old, good_new, intrinsic_mat, np.array(0))

    R = M[1]
    t = M[2]
    # absolute_scale = get_absolute_scale()
    # t = t + absolute_scale * R.dot(t)
    # if absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0]):
    #     t = t + absolute_scale * R.dot(t)
    #     R = R.dot(R)

    return form_transf(R, np.ndarray.flatten(t))


def get_absolute_scale(prev_pose, cur_pose):
    x_prev = float(prev_pose[0, 3])
    y_prev = float(prev_pose[1, 3])
    z_prev = float(prev_pose[2, 3])
    x = float(cur_pose[0, 3])
    y = float(cur_pose[1, 3])
    z = float(cur_pose[2, 3])

    cur_vect = np.array([[x], [y], [z]])
    prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
    return np.linalg.norm(cur_vect - prev_vect)


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
        surf = cv.xfeatures2d.SURF_create(400)
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict()
        flann = cv.FlannBasedMatcher(index_params, search_params)
        # bf = cv.BFMatcher()
        return surf, flann
    elif detector_name == 'orb':
        orb = cv.ORB_create(4000)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict()
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

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat, cv.RANSAC)

    M = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat)

    t = M[2]
    R = M[1]

    return form_transf(R, np.ndarray.flatten(t))


def get_R_T(pts1, pts2, intrinsic_mat):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat, cv.RANSAC)

    M = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat)

    t = M[2]
    R = M[1]

    return R, t


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
        if m.distance < 0.2 * n.distance:
            matchesMask[i] = [1, 0]

    good = []
    for m, n in matches:
        if m.distance < 0.18 * n.distance:
            good.append([m])
    # draw_params = dict(matchColor = (0, 255, 0),
    #                    singlePointColor = (255, 0, 0),
    #                    matchesMask = matchesMask,
    #                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img3 = cv.drawMatchesKnn(img1, keypoint1, img2, keypoint2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3, ), plt.show()


if __name__ == '__main__':
    img1 = cv.imread('./sample/dataset/00/image_0/000000.png')
    img2 = cv.imread('./sample/dataset/00/image_0/000001.png')
    draw_matches(img1, img2)
    # sys.exit(test_find_matches())


def optical_flow(detector, img1, img2):
    kp1 = detector.detect(img1)
    kp1 = np.array([x.pt for x in kp1], dtype=np.float32)
    # lk_params = dict(winSize=(21, 21),
    #                  # maxLevel = 3,
    #                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
    kp2, st, err = cv.calcOpticalFlowPyrLK(img1, img2, kp1, None)  # shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = kp1[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


# def estimate_path(images, detector_name, intrinsic_mat, initial_pose, ground_truth, threshold=1):
#     current_pose = initial_pose
#     track = [current_pose]
#     detector, matcher = create_detector_and_matcher(detector_name)
#     counter_matches = 0
#     camera_rot = np.eye(3)
#     bar = Bar('Обработка изображений:', max=len(images))
#     for i, image in enumerate(images):
#         bar.next()
#         if i == 0:
#             continue
#         else:
#             pts1, pts2 = find_matches(images[i-1], images[i], detector, matcher, threshold)
#             counter_matches += len(pts1)
#             R, T = get_R_T(pts1,pts2,intrinsic_mat)
#             absolute_scale = get_absolute_scale(ground_truth[i - 1], ground_truth[i])
#             print(absolute_scale*camera_rot.dot(T))
#             current_pose = current_pose + absolute_scale*camera_rot.dot(T)
#             camera_rot = R.dot(camera_rot)
#             track.append(np.array([current_pose[0,0], current_pose[1,0], current_pose[2,0]]))
#     bar.finish()
#     print(track)
#     print('Среднее количество "хороших" совпадений: ', counter_matches/len(images))
#     return track

