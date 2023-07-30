from algo import Imitator
import torch, json
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

'''
Trains an Imitator model on human data. 
Saves the model to ./models/imitator.pt
'''

class GermDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


with open('./data/human.json') as f:
    dic = json.load(f)
    episodes = dic['episodes']

X, y = [], []
for episode in episodes:
    if len(episode['states']) < 200:
        continue
    x = episode['states']
    x = [state + ([[-100, -100]] * (5 - len(state))) for state in x]
    X.append(torch.tensor(x))
    y.append(torch.tensor(episode['actions']))
X = torch.cat(X)
y = torch.hstack(y)

print('legal datapoints:', len(y))

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
                torch.save(model, './model/imitator_new.pt')
    
plt.plot(list(range(0, len(losses) * 10, 10)), losses, label='test loss')
plt.plot(list(range(0, len(train_losses))), train_losses, label='train loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./model/new_loss.png')
print('best loss:', best_loss)