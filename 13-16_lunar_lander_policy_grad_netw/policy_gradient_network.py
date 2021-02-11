import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PGN(nn.Module):
    def __init__(self, state_space, n_actions, lr, checkpoint_file):
        super(PGN, self).__init__()

        self.input_layer = nn.Linear(*state_space, 128)  # * means we are going to unpack a list
        self.hid_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = checkpoint_file

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hid_layer(x))
        x = self.output_layer(x)
        return x

    def save_checkpoint(self):
        print('.........Saving checkpoint.........')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.........Loading checkpoint.........')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))




