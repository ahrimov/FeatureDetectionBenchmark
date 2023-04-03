import cv2 as cv
import os
import numpy as np


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images


# Here, i ∈ {0,1,2,3} is the camera index,
# where 0 represents the left grayscale, 1 the right grayscale,
# 2 the left color and 3 the right color camera.

# read only one (left) camera
def load_calib(filename):
    with open(filename, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        projection_mat = np.reshape(params, (3, 4))
        intrinsic_mat = projection_mat[0:3, 0:3]
    return projection_mat, intrinsic_mat


def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses
