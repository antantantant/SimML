# TODO: develop and validate models for these cases
# TODO: visualize 2d data on toy cases (stochastic, hybrid)
# TODO: develop and validate models for these cases

import os
import argparse
import time
import numpy as np
import torch
from torch import nn
import utils
from utils import RunningAverageMeter, accuracy
from simulations import SwitchDataset
import matplotlib
matplotlib.use("TkAgg")
from models import NODEfunc, ODEBlock

parser = argparse.ArgumentParser()
parser.add_argument('--physics', type=str, choices=['switch'], default='switch')  # choose physics model
parser.add_argument('--time_span', type=float, default=1.)  # time span for simulation
parser.add_argument('--dt', type=float, default=0.001)  # time step for simulation
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')  # choose surrogate model
parser.add_argument('--tol', type=float, default=1e-3)  # tolerance for ode solver
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])  # method for computing gradient
parser.add_argument('--nepochs', type=int, default=100)  # number of training epochs
parser.add_argument('--lr', type=float, default=0.1)  # learning rate
parser.add_argument('--batch_size', type=int, default=20)  # batch size for training
parser.add_argument('--test_batch_size', type=int, default=20)  # batch size for validation and test
parser.add_argument('--save', type=str, default='./experiment1')  # save dir
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if __name__ == '__main__':

    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # visualize the model
    def create_dataset(model_type):
        # create simulation dataset
        return {
            # add new models here
            'switch': SwitchDataset(device, args.time_span, args.test_batch_size, args.dt, args.tol, args.tol)
        }[model_type]

    physics_simulation = create_dataset(args.physics)
    # vis = utils.Visualization(dataset=physics_simulation)
    # vis.plot_data()

    # train a baseline Neural ODE model
    # NOTE: please distinguish between physical and surrogate (statistical) models
    settings = {'odefunc': NODEfunc(2),
                'device': device,
                'rtol': args.tol,
                'atol': args.tol}
    surrogate_model = ODEBlock(settings).to(device)

    # save model info
    logger.info(surrogate_model)
    logger.info('Number of parameters: {}'.format(utils.count_parameters(surrogate_model)))

    # define loss
    criterion = nn.MSELoss().to(device)  # TODO: check loss definition

    # get data streamer
    train_loader, test_loader, train_eval_loader = utils.get_data_loaders(physics_simulation, args.batch_size, args.test_batch_size)
    data_gen = utils.inf_generator(train_loader)
    batches_per_epoch = int(args.test_batch_size / args.batch_size)

    # training process
#####################################################################
    # TODO: need to fine-tune the learning rate scheme?
    lr_fn = utils.learning_rate_with_decay(
        args.lr, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    # set up the optimizer
    optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = utils.RunningAverageMeter()
    f_nfe_meter = utils.RunningAverageMeter()
    b_nfe_meter = utils.RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x = data_gen.__next__()
        x0 = x[:,:,0]  # get the initial states
        x0 = x0.to(device)
        pred = surrogate_model(x0.float(), args.dt)  # output is batch_size * dim * time, excluding the initial state
        # compute MSE between physical and surrogate model
        loss = criterion(pred, x[:,:,1:].float())  # exclude initial states since they are the same for pred and target

        # monitor forward ODE steps
        nfe_forward = surrogate_model.nfe
        surrogate_model.nfe = 0

        # compute gradient and do gradient descent
        loss.backward()
        optimizer.step()

        # monitor adjoint steps
        nfe_backward = surrogate_model.nfe
        surrogate_model.nfe = 0

        batch_time_meter.update(time.time() - end)

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)

        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(surrogate_model, args.dt, criterion, train_eval_loader, device)
                val_acc = accuracy(surrogate_model, args.dt, criterion, test_loader, device)
                if val_acc > best_acc:
                    torch.save({'state_dict': surrogate_model.state_dict(), 'args': args},
                               os.path.join(args.save, 'surrogate_model.pth'))
                    best_acc = val_acc
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

    dim, time_step = physics_simulation.dim()
    predict = np.empty((0, dim, time_step))
    for x in test_loader:
        target = x[:,:,1:].to(device).float()
        x0 = x[:,:,0].to(device)
        x = surrogate_model(x0.float(), args.dt).cpu().detach().numpy()
        x0 = np.expand_dims(x0.cpu().detach().numpy(), axis=2)
        x = np.concatenate((x0, x), axis=2)  # add the initial state back in
        predict = np.concatenate((predict, x), axis=0)

    vis = utils.Visualization(dataset=[physics_simulation, predict])
    vis.compare_data()
