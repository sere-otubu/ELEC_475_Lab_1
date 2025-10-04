

import torch
import torch.nn.functional as F
import torch.nn as nn


class autoencoderMLP4Layer(nn.Module):


    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = 392

        # encoder
        self.fc1 = nn.Linear(N_input, N2)
        self.fc2 = nn.Linear(N2, N_bottleneck)

        # decoder
        self.fc3 = nn.Linear(N_bottleneck, N2)
        self.fc4 = nn.Linear(N2, N_output)

        # (optional metadata like in slide)
        self.type = 'MLP4'
        self.input_shape = (1, 28 * 28)

    def encode (self, X):
        # encoder
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return X
    def decode (self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)

        return X

    def forward(self, X):
        return self.decode(self.encode(X))




