# all simulation models are defined here
# use pytorch Dataset class to unify data generation process

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# use direct autodiff if the ode chain is short:
# from torchdiffeq import odeint

# use adjoint method if the ode chain is long:
from torchdiffeq import odeint_adjoint as odeint

from models import LinearODEfunc, SwitchBlock

# single particle, deterministic, 2D, switch between two ODE models
class SwitchDataset(Dataset):
    def __init__(self, *args):
        # TODO:
        device, time_span, sample_size, dt = args
        self.time_span = time_span  # time span
        self.sample_size = sample_size
        self.x0 = np.random.random((sample_size, 4))  # initial state in 2D space (speed and velocity)
        self.dt = dt

        A1 = [[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        A2 = [[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 0.8, 0.2], [0., 0., 0.2, 0.8]]
        odefunc1 = LinearODEfunc(A1)
        odefunc2 = LinearODEfunc(A2)
        domain1 = [[0., 0.5], [0., 1.]]  # [[xmin, xmax], [ymin, ymax]]
        domain2 = [[0.5, 1.0], [0., 1.]]
        settings = {'odefunc1': odefunc1,
                    'domain1': domain1,
                    'odefunc2': odefunc2,
                    'domain2': domain2,
                    'time_span': time_span,
                    'dt': dt,
                    'device': device}
        model = SwitchBlock(settings)
        self.X = model.simulate(self.x0, dt)

    def __len__(self):
        return len(self.sample_size)

    def __getitem__(self, idx):
        return self.X[idx]


def get_data_loaders(batch_size=128, test_batch_size=128):

    trj_data = SwitchDataset(test_batch_size)

    train_loader = DataLoader(dataset=trj_data, batch_size=batch_size, shuffle=False)

    train_eval_loader = DataLoader(dataset=trj_data, batch_size=test_batch_size, shuffle=False)

    test_loader = DataLoader(dataset=trj_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_eval_loader