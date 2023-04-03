import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import loadData


def calculate_track(images, detector, projection_mat, intrinsic_mat, initial_pose):
    current_pose = initial_pose
    track = [current_pose]
    for i, image in enumerate(images):
        if i == 0:
            continue
        else:
            pts1, pts2 = find_matches_sift(images[i-1], images[i])
            transform_mat = get_transform_mat(pts1, pts2, intrinsic_mat)
            current_pose = np.matmul(current_pose, np.linalg.inv(transform_mat))
            print(current_pose)
            track.append(current_pose)
    return track


def find_matches_sift(img1, img2, threshold=0.8):
    sift = cv.SIFT_create()
    keypoint1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoint2, descriptor2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict()

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    # -- Filter matches using the Lowe's ratio test
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < threshold*n.distance:
            pts2.append(keypoint2[m.trainIdx].pt)
            pts1.append(keypoint1[m.queryIdx].pt)

    return pts1, pts2

    # draw_params = dict(matchColor = (0, 255, 0),
    #                    singlePointColor = (255, 0, 0),
    #                    matchesMask = matches_mask,
    #                    flags = cv.DrawMatchesFlags_DEFAULT)
    #
    # img3 = cv.drawMatchesKnn(img1, keypoint1, img2, keypoint2, matches, None, **draw_params)
    # plt.imshow(img3, ), plt.show()

def get_transform_mat(pts1, pts2, intrinsic_mat):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat)

    R = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat, np.array(0))

    t = R[2]
    R = R[1]

    return _form_transf(R, np.ndarray.flatten(t))


def test_find_matches():
    img1 = cv.imread('./sample/dataset/00/image_0/000000.png')
    img2 = cv.imread('./sample/dataset/00/image_0/000001.png')

    pts1, pts2 = find_matches_sift(img1, img2)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    projection_mat, intrinsic_mat = loadData.load_calib('./sample/dataset/00/calib.txt')

    essential_mat, mask = cv.findEssentialMat(pts1, pts2, intrinsic_mat)

    R = cv.recoverPose(essential_mat, pts1, pts2, intrinsic_mat, np.array(0))
    # print(R[0])
    # print(R[1])
    # print(R[2])

    t = R[2]
    R = R[1]

    return _form_transf(R, np.ndarray.flatten(t))

    # print(get_rotation_and_translate_mat(essential_mat, intrinsic_mat, projection_mat, pts1, pts2))


def get_rotation_and_translate_mat(essential_mat, intrinsic_mat, projection_mat, pts1, pts2):
    R1, R2, t = cv.decomposeEssentialMat(essential_mat)

    T1 = _form_transf(R1, np.ndarray.flatten(t))
    T2 = _form_transf(R2, np.ndarray.flatten(t))
    T3 = _form_transf(R1, np.ndarray.flatten(-t))
    T4 = _form_transf(R2, np.ndarray.flatten(-t))
    transformations = [T1, T2, T3, T4]

    # Homogenize K
    K = np.concatenate((intrinsic_mat, np.zeros((3, 1))), axis=1)

    # List of projections
    projections = [K @ T1, K @ T2, K @ T3, K @ T4]

    # np.set_printoptions(suppress=True)

    # print ("\nTransform 1\n" +  str(T1))
    # print ("\nTransform 2\n" +  str(T2))
    # print ("\nTransform 3\n" +  str(T3))
    # print ("\nTransform 4\n" +  str(T4))

    positives = []
    for P, T in zip(projections, transformations):
        hom_Q1 = cv.triangulatePoints(projection_mat, P, pts1.T, pts2.T)
        hom_Q2 = T @ hom_Q1
        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
        relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                 np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
        positives.append(total_sum + relative_scale)

    # Decompose the Essential matrix using built in OpenCV function
    # Form the 4 possible transformation matrix T from R1, R2, and t
    # Create projection matrix using each T, and triangulate points hom_Q1
    # Transform hom_Q1 to second camera using T to create hom_Q2
    # Count how many points in hom_Q1 and hom_Q2 with positive z value
    # Return R and t pair which resulted in the most points with positive z

    max = np.argmax(positives)
    if (max == 2):
        # print(-t)
        return R1, np.ndarray.flatten(-t)
    elif (max == 3):
        # print(-t)
        return R2, np.ndarray.flatten(-t)
    elif (max == 0):
        # print(t)
        return R1, np.ndarray.flatten(t)
    elif (max == 1):
        # print(t)
        return R2, np.ndarray.flatten(t)

def _form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector
    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector
    Returns
    -------
    T (ndarray): The transformation matrix
    """
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
