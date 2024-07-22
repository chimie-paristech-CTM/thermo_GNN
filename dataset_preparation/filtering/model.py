import torch
from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim,  out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)

        return x