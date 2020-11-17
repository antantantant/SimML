# all computational models are defined here

# misc imports (including pytorch)
import numpy as np
import torch
import torch.nn as nn

# use direct autodiff if the ode chain is short:
# from torchdiffeq import odeint

# use adjoint method if the ode chain is long:
from torchdiffeq import odeint_adjoint as odeint


# this is the ODE block
class ODEBlock(nn.Module):

    # initialize settings
    def __init__(self, settings):
        super(ODEBlock, self).__init__()  # use inherited settings
        # various settings for odefunc params
        self.odefunc = settings['odefunc']
        self.device = settings['device']
        self.rtol = settings['rtol']
        self.atol = settings['atol']
        self.integration_time = torch.tensor([0, 1]).float()  # integration time tensor

    # forward iteration specs
    def forward(self, x, dt):
        # need to treat all intermediate ode steps as part of the graph
        batch_size, dim = x.shape  # initialize batch, dim sizes
        out_seq = torch.empty(batch_size, dim, 0)  # output sequence defined as tensor
        n_step = int(1. / dt)  # current step value
        integration_time = torch.tensor([0, dt]).float()  # initialize integration_time
        # loop to iterate through values in n_step (number of steps)
        for i in range(n_step):
            x_new = odeint(self.odefunc, x, integration_time, rtol=self.rtol, atol=self.atol)[1]  # x_new tensor
            x = x_new  # x tensor = prior x_new tensor
            out_seq = torch.cat((out_seq, x_new.unsqueeze(-1)), 2)  # dim 1: batch, dim 2: state dim, dim 3: time
        return out_seq  # output is out_seq

    # step fn specs
    def step(self, x, dt):
        integration_time = torch.tensor([0, dt]).float()  # initialze integration_time
        x_new = odeint(self.odefunc, x, integration_time, rtol=self.rtol, atol=self.atol)[
            1].cpu().detach().numpy()  # x_new tensor
        return x_new  # output is x_new

    # simulate fn specs
    def simulate(self, x, dt):
        out_seq = np.expand_dims(x.cpu().detach().numpy(), axis=2)  # add dimension
        n_step = int(1. / dt)  # initialze n_step
        # loop to iterate through values in n_step
        for i in range(n_step):
            x_new = self.step(x, dt)  # initialize x_new
            x = torch.tensor(x_new).to(self.device).float()  # x tensor
            x_new = np.expand_dims(x_new, axis=2)  # add time as the 3rd dimension
            out_seq = np.concatenate((out_seq, x_new), axis=2)  # dim 1: batch, dim 2: state dim, dim 3: time
        return out_seq  # output is out_seq

    @property  # nfe decorator
    # nfe fn
    def nfe(self):
        return self.odefunc.nfe  # output odefn, nfe

    @nfe.setter  # nfe decorator
    # nfe fn overriding
    def nfe(self, value):
        self.odefunc.nfe = value  # overwrite odefn, nfe as value param


# switch model: single particle, deterministic, 2D, switch between two ODE models
class SwitchBlock(nn.Module):

    # initialize with params
    def __init__(self, settings):
        super(SwitchBlock, self).__init__()  # call inherited settings
        # various ode specss
        self.odefunc1 = settings['odefunc1']
        self.odefunc2 = settings['odefunc2']
        self.domain1 = settings['domain1']
        self.domain2 = settings['domain2']
        self.device = settings['device']
        self.rtol = settings['rtol']
        self.atol = settings['atol']

    # forwards fn (NEED TO FINISH)
    def forward(self, x):
        # TODO: develop a differentiable forward model

        return

    # step fn
    def step(self, x, dt):
        integration_time = torch.tensor([0, dt]).float()  # initialize integration_time
        ind = self.region(x)  # index
        if x[ind].shape[0] > 0:  # if not empty
            x[ind] = odeint(self.odefunc1, x[ind], integration_time, rtol=self.rtol, atol=self.atol)[
                1]  # load specific odeint
        ind = torch.logical_not(ind) + 1  # flip indices
        if x[ind].shape[0] > 0:  # if not empty
            x[ind] = odeint(self.odefunc2, x[ind], integration_time, rtol=self.rtol, atol=self.atol)[1]
        return x.cpu().detach().numpy(), ind.cpu().detach().numpy()  # TODO: check if return value or graph

    # simulate fn
    def simulate(self, x, dt):
        out_seq = np.expand_dims(x.cpu().detach().numpy(), axis=2)  # add dimension to out_seq
        out_ind_seq = self.region(x).cpu().detach().numpy()  # define out_ind_seq as np vector
        n_step = int(1. / dt)  # initialize n_step
        # loop through values in n_step
        for i in range(n_step):
            x_new, ind = self.step(x, dt)  # x_new and ind as output of step fn
            x = torch.tensor(x_new).to(self.device)  # x tensor
            x_new = np.expand_dims(x_new, axis=2)  # add time as the 3rd dimension
            out_seq = np.concatenate((out_seq, x_new), axis=2)  # dim 1: batch, dim 2: state dim, dim 3: time
            out_ind_seq = np.vstack((out_ind_seq, ind))  # to be transposed to: dim 1: batch, dim 2: time
        return out_seq, out_ind_seq.T  # outputs are out_seq, transposed out_ind_seq

    @staticmethod  # region decorator
    # region fn
    def region(x):
        return x[:, 0] < 0.5  # TODO: formalize this

    @property  # nfe decorator
    def nfe(self):
        return self.odefunc.nfe  # return odefunc's nfe

    @nfe.setter  # nfe decorator
    def nfe(self, value):
        self.odefunc.nfe = value  # set odefunc's nfe as input value param


# define physically explainable ODE module
class LinearODEfunc(nn.Module):

    # TODO: odeint with control inputs?
    # initializing odefunc
    def __init__(self, A):
        super(LinearODEfunc, self).__init__()  # call inherited settings
        self.A = torch.tensor(A).float()  # first dim is batch, need float for matrix mul
        self.nfe = 0  # nfe initialized to 0

    # forward fn
    def forward(self, t, x):
        self.nfe += 1  # increment nfe
        out = torch.mm(self.A, torch.transpose(x.float(), 0, 1))  # matrix multiplication of A, transposed x tensor
        return torch.transpose(out, 0, 1).double()  # output is out tensor


# define NN model to mimic ODE
class NODEfunc(nn.Module):

    # this model can be customized
    # initializing odefunc
    def __init__(self, dim):
        super(NODEfunc, self).__init__()  # call inherited settings
        self.linear1 = nn.Linear(dim, 16)  # input layer (1)
        self.linear2 = nn.Linear(16, 32)  # intermediate layer (2)
        self.linear3 = nn.Linear(32, dim)  # output layer (3)
        self.relu = nn.ReLU(inplace=True)  # rectified linear unit
        self.nfe = 0  # initialize nfe

    # forward fn
    def forward(self, t, x):
        self.nfe += 1  # increment nfe
        # iterative outputs using NODE layers (3 sets)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out  # output NODE result
