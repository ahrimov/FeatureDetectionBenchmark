import numpy as np
from matplotlib import pyplot as plt
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
    ax.plot(x, y, linewidth=2.5, color='r', label='Истинный путь')
    for alg in dict:
        estimated_track = dict[alg]
        track = np.array(estimated_track)
        a = track.T
        x = a[0]
        y = a[1]
        ax.plot(x, y, linewidth=1, label=alg)
    ax.legend()
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'track.png')
    plt.show()

    fig, ax = plt.subplots()
    plt.title('График точности вычисленного пути')
    plt.xlabel('Изображения')
    plt.ylabel('Ошибка')
    for alg in dict:
        estimated_track = dict[alg]
        if len(ground_track) == len(estimated_track):
            error = np.linalg.norm(np.array(ground_track) - np.array(estimated_track), axis=1)
            ax.plot(range(len(error)), error, linewidth=1, label=alg)
    ax.legend()
    if output_directory is not None:
        plt.savefig('./' + output_directory + 'error.png')
    plt.show()


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
