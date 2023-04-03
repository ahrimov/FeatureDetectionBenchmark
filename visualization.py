import numpy as np
from matplotlib import pyplot as plt


def show_track(gt_track, es_track):
    track = np.array(gt_track)
    a = track.T
    print(a)
    x = a[0]
    y = a[1]
    print(x)
    print(y)

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.5)

    track = np.array(es_track)
    a = track.T
    print(a)
    x = a[0]
    y = a[1]
    print(x)
    print(y)

    ax.plot(x, y, linewidth=1)

    plt.show()
