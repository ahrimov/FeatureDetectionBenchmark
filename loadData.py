import cv2 as cv
import os
import numpy as np
from progress.bar import Bar


def load_images(folder, st_image=0, end_image=0, count=1):
    images = []
    filenames = os.listdir(folder)
    max_images = len(filenames)
    if st_image == end_image:
        bar = Bar('Загрузка изображений: ', max=max_images)
        for filename in os.listdir(folder):
            image = cv.imread(os.path.join(folder, filename))
            bar.next()
            if image is not None:
                images.append(image)
        bar.finish()
    else:
        if st_image > max_images or end_image > max_images or count > max_images or end_image < st_image:
            return []
        bar = Bar('Загрузка изображений: ', max=(end_image-st_image)/count)
        i = st_image
        while i < end_image:
            filename = filenames[i]
            image = cv.imread(os.path.join(folder, filename))
            bar.next()
            if image is not None:
                images.append(image)
            i += count
        bar.finish()
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


def load_quaternion_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            L = np.fromstring(line, dtype=np.float64, sep=',')
            t = L[1:4]
            quaternion = L[4:8]
            R = quaternion_to_mat(quaternion[0],quaternion[1],quaternion[2],quaternion[3])
            T = np.array([[R[0,0],R[0,1],R[0,2],t[0]],
                          [R[1,0],R[1,1],R[1,2],t[1]],
                          [R[2,0],R[2,1],R[2,2],t[2]],
                          [0, 0, 0, 1]
                          ])
            poses.append(T)
    return poses


def quaternion_to_mat(W, X, Y, Z):
    xx = X * X
    xy = X * Y
    xz = X * Z
    xw = X * W

    yy = Y * Y
    yz = Y * Z
    yw = Y * W

    zz = Z * Z
    zw = Z * W
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)

    m10 = 2 * (xy + zw)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - xw)

    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = 1 - 2 * (xx + yy)
    return np.array([[m00,m01,m02],
                     [m10,m11,m12],
                     [m20,m21,m22]])