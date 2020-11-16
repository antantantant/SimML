# all utility classes here

# misc imports
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging
from torch.utils.data import DataLoader
from torch import nn

# visualizations
class Visualization:
    # initialize viz
    def __init__(self, dataset):
        self.dataset = dataset # initialize dataset
        self.compare = len(dataset) == 2 # initialize compare as bool

    # compare two datasets
    def compare_data(self):
        assert self.compare # test self.compare
        dataset1 = self.dataset[0] # initialize dataset1 as first dim of dataset
        dataset2 = self.dataset[1] # initialize dataset2 as second dim of dataset
        trj1 = dataset1.X # trajectory 1 is X tensor of dataset1
        phase1 = dataset1.phase # phase 1 is phase of dataset1
        trj2 = dataset2  # TODO: right now Neural ODE model does not output phase info, dataset2 is just an array

        fig = plt.figure() # initial plot
        # specify plot axes
        ax = plt.axes(xlim=(dataset1.domain[0][0], dataset1.domain[0][1]),
                      ylim=(dataset1.domain[1][0], dataset1.domain[1][1]))

        # placeholders for the first dataset
        lines = [ax.plot([], [], '-k')[0] for i in range(trj1.shape[0])] # each line plotted as continuous line
        lobj_marker1 = ax.plot([], [], 'or')[0] # dot markers in red
        lobj_marker2 = ax.plot([], [], 'ob')[0] # dot markers in blue
        lines.append(lobj_marker1) # add markers
        lines.append(lobj_marker2) # add markers

        # placeholders for the second dataset
        lines2 = [ax.plot([], [], '-.k')[0] for i in range(trj2.shape[0])] # each line plotted as continuous line
        # lobj_marker1 = ax.plot([], [], 'pr')[0]
        # lobj_marker2 = ax.plot([], [], 'pb')[0]
        # lines2.append(lobj_marker1)
        # lines2.append(lobj_marker2)

        # combine
        lines.extend(lines2)

        # plt draw initializing
        def init():
            # iterate through available lines
            for line in lines:
                line.set_data([], []) # null data
            return lines # output modified lines

        # define drawing (animate) fn
        def animate(i):
            xdata1 = trj1[:, 0, :i] # coordinate x1
            ydata1 = trj1[:, 1, :i] # coordinate y1
            xdata2 = trj2[:, 0, :i] # coordinate x2
            ydata2 = trj2[:, 1, :i] # coordinate y2

            # loop through values in trj1 dim 0
            for j in range(trj1.shape[0]):
                lines[j].set_data(xdata1[j], ydata1[j])  # set data for each line separately.
            lines[trj1.shape[0]].set_data(trj1[phase1[:,i-1]==0, 0, i - 1], trj1[phase1[:,i-1]==0, 1, i - 1]) # specify data for dim 0
            lines[trj1.shape[0]+1].set_data(trj1[phase1[:,i-1]==1, 0, i - 1], trj1[phase1[:,i-1]==1, 1, i - 1]) # specify data for dim 1

            # loop through values in trj2 dim 0
            for j in range(trj2.shape[0]):
                lines[trj1.shape[0]+2+j].set_data(xdata2[j], ydata2[j])  # set data for each line separately.
            # lines[-2].set_data(trj2[phase2[:,i-1]==0, 0, i - 1], trj2[phase2[:,i-1]==0, 1, i - 1])
            # lines[-1].set_data(trj2[phase2[:,i-1]==1, 0, i - 1], trj2[phase2[:,i-1]==1, 1, i - 1])

            return lines # output modified lines

        interval = 10 # specify intervals
        # specs for plt animation
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=trj1.shape[2], interval=interval, blit=True)
        plt.show() # output modified plt

    # plot simulation data from dataset
    def plot_data(self):
        # if self.compare bool is true
        if self.compare:
            dataset = self.dataset[0] # dataset is dim 0 of self
        trj = dataset.X # initialize trj
        phase = dataset.phase # initialize phase
        fig = plt.figure() # plot figure
        # specify axes (positive)
        ax = plt.axes(xlim=(dataset.domain[0][0], dataset.domain[0][1]),
                      ylim=(dataset.domain[1][0], dataset.domain[1][1]))
        lines = [ax.plot([], [], '-k')[0] for i in range(trj.shape[0])] # specify lines
        lobj_marker1 = ax.plot([], [], 'or')[0] # markers
        lobj_marker2 = ax.plot([], [], 'ob')[0] # markers
        lines.append(lobj_marker1) # append markers
        lines.append(lobj_marker2) # append markers

        # initialize plt
        def init():
            # per line in lines
            for line in lines:
                line.set_data([], []) # null data
            return lines # output modified lines

        # animate lines in plt
        def animate(i):
            xdata = trj[:, 0, :i] # xdata as slice of trj
            ydata = trj[:, 1, :i] # ydata as slice of trj
            # loop thru data in flattened trj vector
            for j in range(trj.shape[0]):
                lines[j].set_data(xdata[j], ydata[j])  # set data for each line separately.
            lines[-2].set_data(trj[phase[:,i-1]==0, 0, i - 1], trj[phase[:,i-1]==0, 1, i - 1]) # data for lines
            lines[-1].set_data(trj[phase[:,i-1]==1, 0, i - 1], trj[phase[:,i-1]==1, 1, i - 1]) # data for lines

            return lines # output modified lines

        interval = 10 # specify intervals
        # animation var
        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=trj.shape[2], interval=interval, blit=True)
        plt.show() # display plt


# makedirs fn to make specific dir for experiment
def makedirs(dirname):
    # if dir dne, make dir with specified name
    if not os.path.exists(dirname):
        os.makedirs(dirname) # make dir

# logger fn to write log of experiment to dir
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger() # initialize logger
    # debug option
    if debug:
        level = logging.DEBUG
    # (normal) info option
    else:
        level = logging.INFO
    logger.setLevel(level) # level of logger specificity
    # save option
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a") # specify how to log
        info_file_handler.setLevel(level) # level of logging
        logger.addHandler(info_file_handler) # handle logging
    # display option
    if displaying:
        console_handler = logging.StreamHandler() # info stream for logging
        console_handler.setLevel(level) # level of logging
        logger.addHandler(console_handler) # handler for logging
    logger.info(filepath) # specify filepath for logging
    # specify how to handle file
    with open(filepath, "r") as f:
        logger.info(f.read()) # read file info
    # loop for files in package_files
    for f in package_files:
        logger.info(f) # get info of file
        # specify read for file
        with open(f, "r") as package_f:
            logger.info(package_f.read()) # read file

    return logger # output logger


# use data loaders to stream data for batch training
def get_data_loaders(data, batch_size=128, test_batch_size=128):

    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False) # initialize train_loader

    train_eval_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False) # initialize train_eval_loader

    test_loader = DataLoader(dataset=data, batch_size=test_batch_size, shuffle=False) # initialize test_loader

    return train_loader, test_loader, train_eval_loader # output the loaders


# count the number of parameters used
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # get sum of parameters that need gradient

# infinite training using DataLoaders
def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__() # call initializing settings for iterable
    # while valid data
    while True:
        # try/catch for errors
        try:
            yield iterator.__next__() # proceed to next possible value
        except StopIteration:
            iterator = iterable.__iter__() # stop iterating


# learning rate fn with decay
def learning_rate_with_decay(lr, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr # specify starting learn rates
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs] # boundaries for training
    vals = [initial_learning_rate * decay for decay in decay_rates] # values for learning rate

    # internal learn rate fn
    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True] # get each learn rate
        i = np.argmax(lt) # define index value as the max learn rate
        return vals[i] # vector of learn rates

    return learning_rate_fn # output learn rates


# create a moving window observer
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    # initializing fn
    def __init__(self, momentum=0.99):
        self.momentum = momentum # set momentum
        self.reset() # reset the fn vals

    # reset fn
    def reset(self):
        self.val = None # null val
        self.avg = 0 # null avg

    # update fn
    def update(self, val):
        # if val null
        if self.val is None:
            self.avg = val # set avg to input param val
        # populated val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum) # avg as fn of other available params
        self.val = val # set val


# define the accuracy, this is the goodness measure
# TODO: goodness defined on physically meaningful features?
def accuracy(model, dt, criterion, dataset_loader, device):
    mse = 0 # initial mse null
    # loop through values in dataset_loader
    for x in dataset_loader:
        target = x[:,:,1:].to(device).float() # set target as (tensor) slice of x vector
        x0 = x[:,:,0].to(device) # set x0 as slice of x vector
        predicted = model(x0.float(), dt) # predicted vector as output of model
        mse += criterion(predicted, target).cpu().detach().numpy() # add criterion vector into mse
    return mse / len(dataset_loader.dataset) # return ratio of mse : length of dataset
