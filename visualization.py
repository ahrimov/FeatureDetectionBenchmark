import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['axes.grid'] = True

def show_track(ground_track, estimated_track):
    track = np.array(ground_track)
    a = track.T
    x = a[0]
    y = a[1]

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, y, linewidth=2.5, color='r')

    track = np.array(estimated_track)
    a = track.T
    x = a[0]
    y = a[1]
    ax1.plot(x, y, linewidth=1, color='b')

    error = np.linalg.norm(np.array(ground_track) - np.array(estimated_track), axis=1)
    ax2.plot(range(len(error)), error, linewidth=1)
    plt.show()
