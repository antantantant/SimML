# all utility classes here

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging
from torch.utils.data import DataLoader

class Visualization:
    def __init__(self, dataset):
        self.dataset = dataset

    # plot simulation data from dataset
    def plot_data(self):
        trj = self.dataset.X
        phase = self.dataset.phase
        fig = plt.figure()
        ax = plt.axes(xlim=(self.dataset.domain[0][0], self.dataset.domain[0][1]),
                      ylim=(self.dataset.domain[1][0], self.dataset.domain[1][1]))
        lines = [ax.plot([], [], '-k')[0] for i in range(trj.shape[0])]
        lobj_marker1 = ax.plot([], [], 'or')[0]
        lobj_marker2 = ax.plot([], [], 'ob')[0]
        lines.append(lobj_marker1)
        lines.append(lobj_marker2)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            # print(i)
            xdata = trj[:, 0, :i]
            ydata = trj[:, 1, :i]
            for j in range(trj.shape[0]):
                lines[j].set_data(xdata[j], ydata[j])  # set data for each line separately.
            lines[-2].set_data(trj[phase[:,i-1]==0, 0, i - 1], trj[phase[:,i-1]==0, 1, i - 1])
            lines[-1].set_data(trj[phase[:,i-1]==1, 0, i - 1], trj[phase[:,i-1]==1, 1, i - 1])

            return lines

        interval = 10
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=trj.shape[2], interval=interval, blit=True)
        plt.show()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# use data loaders to stream data for batch training
def get_data_loaders(data, batch_size=128, test_batch_size=128):

    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    train_eval_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False)

    test_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_eval_loader
