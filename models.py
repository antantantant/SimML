# all computational models are defined here

import numpy as np
import torch
import torch.nn as nn

# use direct autodiff if the ode chain is short:
# from torchdiffeq import odeint

# use adjoint method if the ode chain is long:
from torchdiffeq import odeint_adjoint as odeint


# this is the ODE block
class ODEBlock(nn.Module):

    def __init__(self, settings):
        super(ODEBlock, self).__init__()
        self.odefunc = settings['odefunc']
        self.device = settings['device']
        self.rtol = settings['rtol']
        self.atol = settings['atol']
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x, dt):
        # need to treat all intermediate ode steps as part of the graph
        batch_size, dim = x.shape
        out_seq = torch.empty(batch_size, dim, 0)
        n_step = int(1. / dt)
        integration_time = torch.tensor([0, dt]).float()
        for i in range(n_step):
            x_new = odeint(self.odefunc, x, integration_time, rtol=self.rtol, atol=self.atol)[1]
            x = x_new
            out_seq = torch.cat((out_seq, x_new.unsqueeze(-1)), 2)  # dim 1: batch, dim 2: state dim, dim 3: time
        return out_seq

    def step(self, x, dt):
        integration_time = torch.tensor([0, dt]).float()
        x_new = odeint(self.odefunc, x, integration_time, rtol=self.rtol, atol=self.atol)[1].cpu().detach().numpy()
        return x_new

    def simulate(self, x, dt):
        out_seq = np.expand_dims(x.cpu().detach().numpy(), axis=2)
        n_step = int(1. / dt)
        for i in range(n_step):
            x_new = self.step(x, dt)
            x = torch.tensor(x_new).to(self.device).float()
            x_new = np.expand_dims(x_new, axis=2)  # add time as the 3rd dimension
            out_seq = np.concatenate((out_seq, x_new), axis=2)  # dim 1: batch, dim 2: state dim, dim 3: time
        return out_seq

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


# switch model: single particle, deterministic, 2D, switch between two ODE models
class SwitchBlock(nn.Module):

    def __init__(self, settings):
        super(SwitchBlock, self).__init__()
        self.odefunc1 = settings['odefunc1']
        self.odefunc2 = settings['odefunc2']
        self.domain1 = settings['domain1']
        self.domain2 = settings['domain2']
        self.device = settings['device']
        self.rtol = settings['rtol']
        self.atol = settings['atol']


    def forward(self, x):
        # TODO: develop a differentiable forward model

        return

    def step(self, x, dt):
        integration_time = torch.tensor([0, dt]).float()
        ind = self.region(x)
        if x[ind].shape[0] > 0:   # if not empty
            x[ind] = odeint(self.odefunc1, x[ind], integration_time, rtol=self.rtol, atol=self.atol)[1]
        ind = -ind + 1  # flip indices
        if x[ind].shape[0] > 0:  # if not empty
            x[ind] = odeint(self.odefunc2, x[ind], integration_time, rtol=self.rtol, atol=self.atol)[1]
        return x.cpu().detach().numpy(), ind.cpu().detach().numpy()  # TODO: check if return value or graph

    def simulate(self, x, dt):
        out_seq = np.expand_dims(x.cpu().detach().numpy(), axis=2)
        out_ind_seq = self.region(x).cpu().detach().numpy()
        n_step = int(1. / dt)
        for i in range(n_step):
            x_new, ind = self.step(x, dt)
            x = torch.tensor(x_new).to(self.device)
            x_new = np.expand_dims(x_new, axis=2)  # add time as the 3rd dimension
            out_seq = np.concatenate((out_seq, x_new), axis=2)  # dim 1: batch, dim 2: state dim, dim 3: time
            out_ind_seq = np.vstack((out_ind_seq, ind))  # to be transposed to: dim 1: batch, dim 2: time
        return out_seq, out_ind_seq.T

    @staticmethod
    def region(x):
        return x[:, 0] < 0.5  # TODO: formalize this

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


# define physically explainable ODE module
class LinearODEfunc(nn.Module):

    # TODO: odeint with control inputs?
    def __init__(self, A):
        super(LinearODEfunc, self).__init__()
        self.A = torch.tensor(A).float()  # first dim is batch, need float for matrix mul
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = torch.mm(self.A, torch.transpose(x.float(), 0, 1))
        return torch.transpose(out, 0, 1).double()


# define NN model to mimic ODE
class NODEfunc(nn.Module):

    # this model can be customized
    def __init__(self, dim):
        super(NODEfunc, self).__init__()
        self.linear1 = nn.Linear(dim, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, dim)
        self.relu = nn.ReLU(inplace=True)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out