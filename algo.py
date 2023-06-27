import torch

class Imitator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 32)
        self.b1 = torch.nn.BatchNorm1d(32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 64)
        self.b2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(64, 5)
        self.softmax = torch.nn.Softmax(dim=1)
    
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