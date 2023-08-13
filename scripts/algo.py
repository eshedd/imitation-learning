import torch
import torch.nn as nn


class Imitator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.b2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        x = self.fc1(x)
        x = self.b1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.b2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x



class Reinforcer(nn.Module):
    def __init__(self, in_dim=10, hid_dim=1, out_dim=5, n_layers=2, 
                activation=nn.ReLU()):
        super().__init__()

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(in_dim, hid_dim))
        self.mlp.append(activation)
        for _ in range(n_layers):
            self.mlp.append(nn.Linear(hid_dim, hid_dim))
            self.mlp.append(activation)
        self.mlp.append(nn.Linear(hid_dim, out_dim))
        self.mlp.append(nn.Softmax(dim=1))


    def forward(self, x):
        x = x.reshape(x.shape[0], -1).float()
        for layer in self.mlp:
            x = layer(x)
        return x