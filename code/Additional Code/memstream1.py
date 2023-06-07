import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import metrics
import scipy.spatial as sp
from torch.autograd import Variable
import argparse
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument("--dev", help="device", default="cuda:0")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=2048)
parser.add_argument("--seed", type=int, help="random seed", default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
nfile = None
lfile = None
if args.dataset == 'NSL':
    nfile = 'nsl.txt'
    lfile = 'nsllabel.txt'
elif args.dataset == 'KDD':
    nfile = 'kdd.txt'
    lfile = 'kddlabel.txt'
elif args.dataset == 'UNSW':
    nfile = 'unsw.txt'
    lfile = 'unswlabel.txt'
elif args.dataset == 'DOS':
    nfile = 'dos.txt'
    lfile = 'doslabel.txt'
else:
    df = scipy.io.loadmat(+args.dataset+".mat")
    numeric = torch.FloatTensor(df['X'])
    labels = (df['y']).astype(float).reshape(-1)

device = torch.device(args.dev)

class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim*2
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.mean = None
        self.std = None
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.Tanh(),
        ).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(device)
        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0
        
        # initialize mean and std
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)

    def train_autoencoder(self, data, epochs):
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = Variable(new)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(new + 0.001*torch.randn_like(new).to(device)))
            target = new.view(-1, self.in_dim)  # reshape target tensor
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if loss.item() < 0.0001:
                break

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count%self.memory_len
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = model.mem_data.mean(0), model.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder(new)
        self.memory.requires_grad = False
        self.mem_data = x

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)
        decoder_output = self.decoder(encoder_output)
        return decoder_output, encoder_output


def split_data(numeric, labels, train_size=0.8, val_size=0.1, test_size=0.1, shuffle=True, random_seed=0):
        assert train_size + val_size + test_size == 1
        n_samples = len(numeric)
        indices = np.arange(n_samples)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_end = int(train_size * n_samples)
        val_end = int((train_size + val_size) * n_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = numeric[train_indices]
        val_data = numeric[val_indices]
        test_data = numeric[test_indices]

        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]

        return train_data, train_labels, val_data, val_labels, test_data, test_labels    
    
def train(model, device, train_loader, val_loader, optimizer, criterion, epochs):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, output_loss = model(data)
            loss = criterion(output_loss, torch.zeros_like(output_loss[0]))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(data)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        running_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output, output_loss = model(data)
                loss = criterion(output_loss, torch.zeros_like(output_loss[0]))
                running_loss += loss.item() * len(data)
            val_loss = running_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, train_loss, val_loss))
    return train_losses, val_losses




if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
    labels = np.loadtxt(lfile, delimiter=',')

if args.dataset == 'KDD':
    labels = 1 - labels
torch.manual_seed(args.seed)
N = args.memlen
params = {
          'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
         }

model = MemStream(numeric[0].shape[0],params).to(device)
train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(numeric, labels)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
#data_loader = DataLoader(numeric, batch_size=batch_size)
train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(list(zip(val_data, val_labels)), batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=params['batch_size'], shuffle=True)

train_loss, val_loss = train(model, device, train_loader, val_loader, model.optimizer, model.loss_fn, args.epochs)

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()






# init_data = numeric[labels == 0][:N].to(device)
# model.mem_data = init_data
# torch.set_grad_enabled(True)
# model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
# torch.set_grad_enabled(False)
# model.initialize_memory(Variable(init_data[:N]))
# err = []
# for data in data_loader:
#     output = model(data.to(device))
#     err.append(output)
#     print("Loss:", output.item())
# scores = np.array([i.cpu() for i in err])
# auc = metrics.roc_auc_score(labels, scores)
# print("ROC-AUC", auc)
