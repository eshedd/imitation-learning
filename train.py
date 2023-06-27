from algo import Imitator
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GermDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


with open('data.txt') as f:
    episodes = [line.rstrip() for line in f]

X, Y = [], []
for episode in episodes:
    y, x = episode[1:].split('|')
    x = eval(x)
    x = [state + ([[-100, -100]] * (4 - len(state))) for state in x]
    X.append(torch.tensor(x))
    moves = ['l', 'r', 'u', 'd', 's']
    Y.append(torch.tensor([moves.index(move) for move in y[:-1]]))
X = torch.cat(X)
y = torch.hstack(Y)

torch.manual_seed(0)

dataset = GermDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

model = Imitator()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = torch.nn.CrossEntropyLoss()

losses = []
train_losses = []
best_loss = 100000
for epoch in range(500):
    model.train()
    t_loss = 0
    i = 0
    for X, y in train_dataloader:
        optimizer.zero_grad()
        y_hat = model(X)
        y = torch.nn.functional.one_hot(y, 5).float()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        i += 1
    train_losses.append(t_loss/i)
    
    with torch.no_grad():
        model.eval()
        if epoch % 10 == 0:
            for X, y in test_dataloader:
                y_hat = model(X)
                y = torch.nn.functional.one_hot(y, 5).float()
                loss = criterion(y_hat, y)
            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model, 'imitator.pt')
    
plt.plot(list(range(0, len(losses) * 10, 10)), losses, label='test loss')
plt.plot(list(range(0, len(train_losses))), train_losses, label='train loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
print('best loss:', best_loss)