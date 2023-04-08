import cv2 as cv
import numpy as np
import random
import sys
from matplotlib import pyplot as plt
from progress.bar import Bar


def add_blur(images):
    res_images = []
    bar = Bar('Добавление размытия:', max=len(images))
    for image in images:
        blur = cv.blur(image, (5,5))
        res_images.append(blur)
        bar.next()
    bar.finish()
    return res_images


def add_sp_noise(images, prob=0.05):
    res_images = []
    bar = Bar('Добавление импульсивного шума:', max=len(images))
    for image in images:
        noised_image = sp_noise(image, prob)
        res_images.append(noised_image)
        bar.next()
    bar.finish()
    return res_images


def add_gaussian_noise(images, prob=0.5):
    res_images = []
    bar = Bar('Добавление шума гауссиана:', max=len(images))
    for image in images:
        noised_image = gaussian_noise(image, prob)
        res_images.append(noised_image)
        bar.next()
    bar.finish()
    return res_images


def add_uniform_noise(images, prob=0.5):
    res_images = []
    bar = Bar('Добавление нормального шума:', max=len(images))
    for image in images:
        noised_image = uniform_noise(image, prob)
        res_images.append(noised_image)
        bar.next()
    bar.finish()
    return res_images


def sp_noise(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gaussian_noise(image, prob):
    gauss_noise = np.zeros(image.shape, dtype=np.uint8)
    cv.randn(gauss_noise, 128, 20)
    gauss_noise = (gauss_noise * prob).astype(np.uint8)
    gn_img = cv.add(image, gauss_noise)
    return gn_img


def uniform_noise(image, prob):
    uni_noise = np.zeros(image.shape, dtype=np.uint8)
    cv.randu(uni_noise, 0, 255)
    uni_noise = (uni_noise * prob).astype(np.uint8)
    un_img = cv.add(image, uni_noise)
    return un_img


def test_gaus():
    image = cv.imread('./sample/dataset/00/image_0/000000.png',0)
    noised_image = gaussian_noise(image, 0.5)
    cv.imshow('image', noised_image)
    cv.waitKey(0)


def test_blur():
    image = cv.imread('./sample/dataset/00/image_0/000000.png')
    blur = cv.blur(image, (5,5))
    plt.subplot(121), plt.imshow(image), plt.title('Original')
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.show()


def test_sp_noise():
    image = cv.imread('./sample/dataset/00/image_0/000000.png',0)
    noised_image = sp_noise(image, 0.01)
    cv.imshow('image', noised_image)
    cv.waitKey(0)


def test_uniform_noise():
    image = cv.imread('./sample/dataset/00/image_0/000000.png',0)
    noised_image = uniform_noise(image, 0.5)
    cv.imshow('image', noised_image)
    cv.waitKey(0)


if __name__ == '__main__':
    sys.exit(test_gaus())
