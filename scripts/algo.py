import torch

class Imitator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 32)
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



class Reinforcer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 64)
        self.b1 = torch.nn.BatchNorm1d(64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 128)
        self.b2 = torch.nn.BatchNorm1d(128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 256)
        self.b3 = torch.nn.BatchNorm1d(256)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(256, 32)
        self.b4 = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(32, 5)
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
        x = self.b3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.b4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x