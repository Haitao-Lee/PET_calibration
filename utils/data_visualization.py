# coding = utf-8
import matplotlib.pyplot as plt
import data_input
import numpy as np
# import data_transform
import data_set
import os
from config import args
import random


def main():
    luts = []
    peaks = []
    for path in os.listdir(args.exp_train_lut):
        luts.append(np.load(args.exp_train_lut + '/' + path))
    for path in os.listdir(args.exp_train_peak):
        peaks.append(np.load(args.exp_train_peak + '/' + path))
    mean = data_set.mean_cordinate[0,:]
    mean.resize([256, 2])
    # x = data_set.mean_cordiate[0, 1::2]
    # y = data_set.mean_cordiate[0, 0::2]
    for i in range(len(luts)):
        #i = i + 600
        _, axes = plt.subplots(nrows=1, ncols=2)
        axes[1].matshow(luts[i],  alpha=1)
        #axes[1].plot(peaks[i][:,1],peaks[i][:,0], "yo")
        tmp = [x for x in range(256)]
        random.shuffle(tmp)
        peaks_tmp = peaks[i]
        peaks_ran = np.zeros(peaks_tmp.shape)
        for k in range(256):
            peaks_ran[k,:] = peaks_tmp[tmp[k], :]
        
        axes[1].plot(peaks[i][0::4,1], peaks[i][0::4,0],"ro")
        axes[1].plot(peaks[i][1::4,1], peaks[i][1::4,0],"yo")
        axes[1].plot(peaks[i][2::4,1], peaks[i][2::4,0],"wo")
        axes[1].plot(peaks[i][3::4,1], peaks[i][3::4,0],"bo")
        
        axes[1].set_title("groundtruth%d"%i)
        axes[0].matshow(luts[i],  alpha=1)
        axes[0].plot(mean[0::4,1], mean[0::4,0],"ro")
        axes[0].plot(mean[1::4,1], mean[1::4,0],"yo")
        axes[0].plot(mean[2::4,1], mean[2::4,0],"wo")
        axes[0].plot(mean[3::4,1], mean[3::4,0],"bo")
        axes[0].plot(mean[::16,1], mean[::16,0],"g*")
        axes[0].set_title("mean")
        plt.show()
    return 0


def image_label_visualization(image, row_peaks, col_peaks, markers = None):
    """
    Display an image with peaks marked and allow keyboard control.

    Parameters:
        image (numpy.ndarray): The image to display.
        row_peaks (list): List of row indices for peaks.
        col_peaks (list): List of column indices for peaks.
    """
    fig, ax = plt.subplots()
    ax.matshow(image, alpha=1)
    ax.plot(col_peaks, row_peaks, "ro", markersize=20)
    if markers is not None:
        markers = np.array(markers).reshape(-1, 2)
        ax.plot(markers[0::4, 1], markers[0::4, 0], "w*", markersize=20)
        ax.plot(markers[1::4, 1], markers[1::4, 0], "g*", markersize=20)
        ax.plot(markers[2::4, 1], markers[2::4, 0], "b*", markersize=20)
        ax.plot(markers[3::4, 1], markers[3::4, 0], "y*", markersize=20)
    ax.set_title("Image with Label")
    plt.show()


if __name__ == '__main__':
    main()
