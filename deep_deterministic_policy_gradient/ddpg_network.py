import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DDPGLowDim(nn.Module):
    def __init__(self, n_states, n_actions, lr, gamma, tau):
        super(DDPGLowDim, self).__init__()

        # layers different from the last should be initialized with 1/f, where f=n_input for that layer
        self.fc1 = nn.Linear(n_states, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(400, n_actions)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)