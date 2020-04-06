# --- Import libraries ---
import gc
import os
import scipy.io.wavfile

import matplotlib.pyplot as plt
import numpy as np


# --- Get the current working directory ---
cwd = os.getcwd()


# --- Change directories ---
os.chdir("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\partitioned")


# --- Save the new directory ---
input_dir = os.getcwd()


# --- Generate oscillograms ---
for r, d, f in os.walk(input_dir):
    for file in f:
        # read wav file as np array (y-axis)
        sample_rate, data = scipy.io.wavfile.read(os.path.join(r, file))

        # x-axis
        time = np.linspace(0., 1., data.shape[0])

        # image size
        fig = plt.figure(figsize=(6, 6))

        # make sure axes not plotted
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plot image
        plt.plot(time, data)

        # save image
        if r.split("\\")[-2] == "validation":
            plt.savefig("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\oscillograms\\validation\\" + r.split("\\")[-1] + "\\" + file[:-4] + ".png")
        elif r.split("\\")[-2] == "test":
            plt.savefig("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\oscillograms\\test\\" + r.split("\\")[-1] + "\\" + file[:-4] + ".png")
        else:
            plt.savefig("C:\\Users\\15713\\Desktop\\DS Projects\\Speech Recognition\\speech-recognition\\data\\processed\\oscillograms\\training\\" + r.split("\\")[-1] + "\\" + file[:-4] + ".png")

        # close stuff
        fig.clear()
        plt.clf()
        plt.close(fig)
        gc.collect()


# --- Change back to the original directory ---
os.chdir(cwd)