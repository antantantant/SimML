# TODO: develop and validate models for these cases
# TODO: visualize 2d data on toy cases (stochastic, hybrid)
# TODO: develop and validate models for these cases

# misc library imports
import os
import argparse
import time
import numpy as np
# pytorch imports
import torch
from torch import nn
# experiment imports
import utils
from utils import RunningAverageMeter, accuracy
# simpy import
from simulations import SwitchDataset
# matplotlib for tkagg (tkinter) fw
import matplotlib
matplotlib.use("TkAgg")
# NODE imports
from models import NODEfunc, ODEBlock
# force FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# setting args to be passed into model
parser = argparse.ArgumentParser()
parser.add_argument('--physics', type=str, choices=['switch'], default='switch')  # choose physics model
parser.add_argument('--time_span', type=float, default=1.)  # time span for simulation
parser.add_argument('--dt', type=float, default=0.001)  # time step for simulation
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')  # choose surrogate model
parser.add_argument('--tol', type=float, default=1e-3)  # tolerance for ode solver
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])  # method for computing gradient
parser.add_argument('--nepochs', type=int, default=8)  # number of training epochs; default = 100
parser.add_argument('--lr', type=float, default=0.1)  # learning rate
parser.add_argument('--batch_size', type=int, default=40)  # batch size for training; default = 20
parser.add_argument('--test_batch_size', type=int, default=40)  # batch size for validation and test; default = 20
parser.add_argument('--save', type=str, default='./experiment1')  # save dir
parser.add_argument('--debug', action='store_true') # save debug output
parser.add_argument('--gpu', type=int, default=0) # activate gpu if available (not default)
args = parser.parse_args() # add all args

# initializing experiment
if __name__ == '__main__':

    utils.makedirs(args.save) # create directory based on args
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__)) # initialize logger, save to file
    logger.info(args) # specify information in logger

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') # specify device for model (CUDA gpu if available)

    # visualize the model
    def create_dataset(model_type):
        # create simulation dataset
        return {
            # add new models here
            'switch': SwitchDataset(device, args.time_span, args.test_batch_size, args.dt, args.tol, args.tol) # initialze model dataset
        }[model_type]

    physics_simulation = create_dataset(args.physics) # call prior fn to generate dataset per args
    # vis = utils.Visualization(dataset=physics_simulation)
    # vis.plot_data()

    # train a baseline Neural ODE model
    # NOTE: please distinguish between physical and surrogate (statistical) models
    settings = {'odefunc': NODEfunc(2), # 2D
                'device': device, # gpu/cpu if available
                'rtol': args.tol, # tolerance
                'atol': args.tol} # tolerance
    surrogate_model = ODEBlock(settings).to(device) # used to approximate outcomes

    # save model info
    logger.info(surrogate_model) # store surrogate info
    logger.info('Number of parameters: {}'.format(utils.count_parameters(surrogate_model))) # store number of parameters from args

    # define loss
    # NOTE: using MSE loss
    criterion = nn.MSELoss().to(device)  # TODO: check loss definition

    # get data streamer
    train_loader, test_loader, train_eval_loader = utils.get_data_loaders(physics_simulation, args.batch_size, args.test_batch_size) # data loaders
    data_gen = utils.inf_generator(train_loader) # generate data based on train split
    batches_per_epoch = int(args.test_batch_size / args.batch_size) # batches/epoch as ratio of test:total batch size

    # training process
#####################################################################
    # TODO: need to fine-tune the learning rate scheme?
    # learning function
    lr_fn = utils.learning_rate_with_decay(
        args.lr, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140], # specify args, batches (by boundary values)
        decay_rates=[1, 0.1, 0.01, 0.001] # specify decay rate
    )

    # set up the optimizer
    optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=args.lr, momentum=0.9) # use SGD as optimization algo

    # params for accuracy
    best_acc = 0 # best accuracy value
    batch_time_meter = utils.RunningAverageMeter() # batch timing
    f_nfe_meter = utils.RunningAverageMeter() # forward epoch accuracy
    b_nfe_meter = utils.RunningAverageMeter() # batch epoch accuracy
    end = time.time() # time duration of model training

    # loop through num. of batches
    for itr in range(args.nepochs * batches_per_epoch):

        # specify learning fn for each param group
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad() # zero-out gradients as initial spec
        x = data_gen.__next__() # x = next iteration within data_gen
        x0 = x[:,:,0]  # get the initial states
        x0 = x0.to(device) # send initial states to gpu/cpu
        pred = surrogate_model(x0.float(), args.dt)  # output is batch_size * dim * time, excluding the initial state
        # compute MSE between physical and surrogate model
        loss = criterion(pred, x[:,:,1:].float())  # exclude initial states since they are the same for pred and target

        # monitor forward ODE steps
        nfe_forward = surrogate_model.nfe # extract features
        surrogate_model.nfe = 0 # initialize to 0

        # compute gradient and do gradient descent
        loss.backward() # accumulate gradient
        optimizer.step() # update current gradient

        # monitor adjoint steps
        nfe_backward = surrogate_model.nfe # extract features
        surrogate_model.nfe = 0 # initialize to 0

        batch_time_meter.update(time.time() - end) # add in elapsed time

        f_nfe_meter.update(nfe_forward) # forward epoch accuracy
        b_nfe_meter.update(nfe_backward) # backward epoch accuracy

        end = time.time() # restate end time

        # conditional: proceed if iterations evenly divisible by batches/epoch
        if itr % batches_per_epoch == 0:
            # inference w/o gradient calc
            with torch.no_grad():
                train_acc = accuracy(surrogate_model, args.dt, criterion, train_eval_loader, device) # training accuracy
                val_acc = accuracy(surrogate_model, args.dt, criterion, test_loader, device) # validation accuracy
                # conditional: if validation beats train accuracy
                if val_acc > best_acc:
                    # save the surrogate_model
                    torch.save({'state_dict': surrogate_model.state_dict(), 'args': args},
                               os.path.join(args.save, 'surrogate_model.pth'))
                    # update best accuracy as validation accuracy
                    best_acc = val_acc
                # save info about current epoch into logger
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Err {:.4f} | Test Err {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
#####################################################################

    # visualize surrogate model along with the physics simulation

    # # TODO: load the model if pre-trained
    # surrogate_model.load_state_dict(torch.load(os.path.join(args.save, 'surrogate_model.pth'))['state_dict'],
    #                                 strict=False)
    # surrogate_model.eval()

    dim, time_step = physics_simulation.dim() # specify the dimension, time step for currogate model
    predict = np.empty((0, dim, time_step)) # prediction vector
    # per value in test data
    for x in test_loader:
        target = x[:,:,1:].to(device).float() # target tensor
        x0 = x[:,:,0].to(device) # x0 tensor
        x = surrogate_model(x0.float(), args.dt).cpu().detach().numpy() # gather x tensor from surrogate model
        x0 = np.expand_dims(x0.cpu().detach().numpy(), axis=2) # add dim as np vector
        x = np.concatenate((x0, x), axis=2)  # add the initial state (x0) as dim of x
        predict = np.concatenate((predict, x), axis=0) # prediction vector

    vis = utils.Visualization(dataset=[physics_simulation, predict]) # visualize the prediction vector
    vis.compare_data() # compare visualization data
