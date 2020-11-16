# all simulation models are defined here
# use pytorch Dataset class to unify data generation process

# misc imports (pytorch + defined models)
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models import LinearODEfunc, SwitchBlock


# single particle, deterministic, 2D, switch between two ODE models
class SwitchDataset(Dataset):

    # initializing SwitchDataset
    def __init__(self, *args):
        device, time_span, sample_size, dt, rtol, atol = args # initialze various args
        self.time_span = time_span  # time span
        self.sample_size = sample_size # sample size

        # initial state in 2D space (speed and velocity)
        np.random.seed(0) # using random seed for random nums
        self.x0 = torch.tensor(np.random.uniform(low=0., high=1., size=(sample_size, 2))) # initialze x0
        self.dt = dt # initialize dt
        self.domain = [[0., 1.], [0., 1.]] # initialize domain as 2d matrix (map from positive cartesian plane)
        # TODO: need to rescale x0 if domain is not [0,1]

        # define two odes
        A1 = [[-1., 1], [0.2, -1.]]  # moving at constant speed
        A2 = [[-0.2, -1.], [-1., 1.]] # inverse of A1
        odefunc1 = LinearODEfunc(A1) # initialize odefunc1
        odefunc2 = LinearODEfunc(A2) # initialize odefunc2
        domain1 = [[0., 0.5], [0., 1.]]  # [[xmin, xmax], [ymin, ymax]]
        domain2 = [[0.5, 1.0], [0., 1.]] # [[xmin, xmax], [ymin, ymax]]
        # various settings
        settings = {'odefunc1': odefunc1,
                    'domain1': domain1,
                    'odefunc2': odefunc2,
                    'domain2': domain2,
                    'time_span': time_span,
                    'dt': dt,
                    'device': device,
                    'rtol': rtol,
                    'atol': atol}
        model = SwitchBlock(settings) # create model with specified settings
        # X is the trajectory, phase is the ode model ID (0 or 1)
        self.X, self.phase = model.simulate(self.x0, dt) # define X, phase as outputs of simulate fn

    # dim fn
    def dim(self):
        return self.X.shape[1], self.X.shape[2]  # dim of the state space, time step

    # (rewrites len, getter) len fn
    def __len__(self):
        return self.sample_size # output sample_size

    # (rewrites getitem, getter) getitem fn
    def __getitem__(self, idx):
        return self.X[idx] # output X[idx]
