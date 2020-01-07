# TODO: develop and validate models for these cases
# TODO: visualize 2d data on toy cases (stochastic, hybrid)
# TODO: develop and validate models for these cases

import os
import argparse
import time
import numpy as np
import torch
import utils
from simulations import SwitchDataset
import matplotlib
matplotlib.use("TkAgg")

parser = argparse.ArgumentParser()
parser.add_argument('--physics', type=str, choices=['switch'], default='switch')  # choose physics model
parser.add_argument('--time_span', type=float, default=1.)  # time span for simulation
parser.add_argument('--dt', type=float, default=0.001)  # time step for simulation
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')  # choose surrogate model
parser.add_argument('--tol', type=float, default=1e-3)  # tolerance for ode solver
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])  # method for computing gradient
parser.add_argument('--nepochs', type=int, default=50)  # number of training epochs
parser.add_argument('--lr', type=float, default=0.1)  # learning rate
parser.add_argument('--batch_size', type=int, default=1)  # batch size for training
parser.add_argument('--test_batch_size', type=int, default=50)  # batch size for validation and test
parser.add_argument('--save', type=str, default='./experiment1')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if __name__ == '__main__':

    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    def create_dataset(model_type):
        # create simulation dataset
        return {
            # add new models here
            'switch': SwitchDataset(device, args.time_span, args.test_batch_size, args.dt, args.tol, args.tol)
        }[model_type]

    physics_simulation = create_dataset(args.physics)
    vis = utils.Visualization(dataset=physics_simulation)
    vis.plot_data()
