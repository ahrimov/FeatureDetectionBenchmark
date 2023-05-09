import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
plt.rcParams['axes.grid'] = True


def show_tracks(ground_track, dict, output_directory=None):
    track = np.array(ground_track)
    a = track.T
    x = a[0]
    y = a[1]

    fig, ax = plt.subplots()
    plt.title('Путь')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot(x, y, linewidth=4, color='r', label='Истинный путь')
    for i, alg in enumerate(dict):
        estimated_track = dict[alg]
        track = np.array(estimated_track)
        a = track.T
        x = a[0]
        y = a[1]
        ax.plot(x, y, linewidth=2, linestyle=get_linestyle(i), label=alg)
    ax.legend(fontsize="large")
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'track.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.title('График точности вычисленного пути')
    plt.xlabel('Изображения')
    plt.ylabel('Ошибка')
    for i, alg in enumerate(dict):
        estimated_track = dict[alg]
        if len(ground_track) == len(estimated_track):
            error = np.linalg.norm(np.array(ground_track) - np.array(estimated_track), axis=1)
            ax.plot(range(len(error)), error, linewidth=2, linestyle=get_linestyle(i), label=alg)
    ax.legend(fontsize="large")
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'error.png')
    plt.show()


def get_linestyle(index = 0):
    linestyles = [
         (0, (1, 1)),
        (5, (10, 3)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]
    return linestyles[index]


def show_track(ground_track, estimated_track, output_directory=None):
    track = np.array(ground_track)
    a = track.T
    x = a[0]
    y = a[1]

    fig, ax = plt.subplots()
    plt.title('Путь')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot(x, y, linewidth=2.5, color='r', label='Истинный путь')

    track = np.array(estimated_track)
    a = track.T
    x = a[0]
    y = a[1]
    ax.plot(x, y, linewidth=1, color='b', label='Вычисленный путь')
    ax.legend()
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'track.png')
    plt.show()

    fig, ax = plt.subplots()
    error = np.linalg.norm(np.array(ground_track) - np.array(estimated_track), axis=1)
    plt.title('График точности вычисленного пути')
    plt.xlabel('Изображения')
    plt.ylabel('Ошибка')
    ax.plot(range(len(error)), error, linewidth=1)
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'error.png')
    plt.show()

if __name__ == '__main__':
    img = cv.imread('./sample/sbp.jpg')
    cv.imshow('img', img)
    cv.waitKey(0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.ORB_create(2000)
    kp = sift.detect(gray,None)
    img = cv.drawKeypoints(gray, kp, img)
    cv.imshow('img',img)
    cv.waitKey(0)