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
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

torch.cuda.empty_cache()

# Confirm that the GPU is detected

assert torch.cuda.is_available()

# Get the GPU device name.
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")


start_time = time.time()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument("--dev", help="device", default="cuda:0")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=2048)
parser.add_argument("--seed", type=int, help="random seed", default=0)
parser.add_argument("--act_fn",help="Activation fucntion", default="Sigmoid")
parser.add_argument("--sp",help="Sampling Method", default="rand")
parser.add_argument("--RQ1", help="Hypertune best metrics : Default False", default=False)
parser.add_argument("--RQ2", help="Effect of Activation Functions : Default False",default=False)
parser.add_argument("--RQ3", help="Memory Poisoning Prevention Analysis",default=False)
parser.add_argument("--RQ4", help="Concept Drift",default=False)
parser.add_argument("--RQ5", help="Impact of Memory",default=False)

args = parser.parse_args()

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
elif args.dataset == 'SYN':
    nfile = 'syn.txt'
    lfile = 'synlabel.txt'
else:
    df = scipy.io.loadmat(args.dataset+".mat")
    numeric = torch.FloatTensor(df['X'])
    labels = (df['y']).astype(float).reshape(-1)

device = torch.device(args.dev)

class MemStream(nn.Module):
    def __init__(self, in_dim, params, act_fn):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim * 2
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        
        self.generator = nn.Sequential(
            nn.Linear(self.in_dim,500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.out_dim),
            nn.Tanh()
        ).to(device)
        
        self.discriminator = nn.Sequential(
            nn.Linear(self.out_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, self.in_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.clock = 0
        self.last_update = -1
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=params['lr'])
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=params['lr'])
        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']
        self.generator_loss_fn = nn.BCELoss()
        self.discriminator_loss_fn = nn.BCELoss()

    def train_generator(self, real_data):
        self.generator_optimizer.zero_grad()
        noise = torch.randn(self.batch_size, self.in_dim).to(device)
        fake_data = self.discriminator(noise)
        validity = self.discriminator(fake_data)
        generator_loss = self.generator_loss_fn(validity, torch.ones(self.batch_size, 1).to(device))
        generator_loss.backward()
        self.generator_optimizer.step()
        return generator_loss.item()

    def train_discriminator(self, real_data):
        self.discriminator_optimizer.zero_grad()
        noise = torch.randn(self.batch_size, self.in_dim).to(device)
        fake_data = self.generator(noise).detach()
        validity_real = self.discriminator(real_data)
        validity_fake = self.discriminator(fake_data)
        discriminator_loss_real = self.discriminator_loss_fn(validity_real, torch.ones(self.batch_size, 1).to(device))
        discriminator_loss_fake = self.discriminator_loss_fn(validity_fake, torch.zeros(self.batch_size, 1).to(device))
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        return discriminator_loss.item()


    def train_autoencoder(self, real_data, epochs):
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            self.generator_loss = []
            self.discriminator_loss = []
            for i in range(0, len(real_data), self.batch_size):
                batch_data = real_data[i:i+self.batch_size]

                # Train Generator
                self.generator_optimizer.zero_grad()
                noise = torch.randn(self.batch_size, self.in_dim).to(device)
                fake_data = self.generator(noise)
                validity = self.discriminator(fake_data)
                generator_loss = self.generator_loss_fn(validity, torch.ones(self.batch_size, 1).to(device).expand_as(validity))

                generator_loss.backward()
                self.generator_optimizer.step()
                self.generator_loss.append(generator_loss.item())

                # Train Discriminator
                self.discriminator_optimizer.zero_grad()
                real_batch_data = batch_data.to(device)
                validity_real = self.generator(real_batch_data)
                validity_fake = self.discriminator(fake_data.detach())
                discriminator_loss_real = self.discriminator_loss_fn(validity_real, torch.ones(self.batch_size, 1).to(device).expand_as(validity_real))
                discriminator_loss_fake = self.discriminator_loss_fn(validity_fake, torch.zeros(self.batch_size, 1).to(device).expand_as(validity_fake))
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                self.discriminator_loss.append(discriminator_loss.item())

            print(f"Epoch: {epoch}, Generator Loss: {sum(self.generator_loss)/len(self.generator_loss)} \n") 
            print(f"Epoch: {epoch}, Discriminator Loss:{sum(self.discriminator_loss)/len(self.discriminator_loss)}")


    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            if least_used_pos < self.memory.size(0):
                self.memory[least_used_pos] = encoder_output.detach()
                self.mem_data[least_used_pos] = data.detach()
                self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
                self.count += 1
                return 1
        return 0

    def initialize_memory(self, x):
        self.mem_data = x.to(device)
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        self.memory = self.generator(new.to(device)).detach()
        self.memory.requires_grad = False

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.generator(self.discriminator(new.to(device)))
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        self.update_memory(loss_values, encoder_output, x.to(device))
        return loss_values




if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
    labels = np.loadtxt(lfile, delimiter=',')
    act_fn = nn.Tanh()
else:
    act_fn = nn.LogSoftmax(dim=1)

if args.dataset == 'KDD':
    labels = 1 - labels

torch.manual_seed(args.seed)
N = args.memlen
params = {
          'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr,
          'num_epochs': args.epochs
         }

if args.dataset == 'SYN':
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ',')).reshape(-1, 1)
    labels = np.loadtxt(lfile, delimiter=',')

model = MemStream(numeric[0].shape[0], params, act_fn).to(device)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.mem_data = init_data
torch.set_grad_enabled(True)
model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
torch.set_grad_enabled(False)
model.initialize_memory(Variable(init_data[:N]))

for epoch in range(args.epochs):
    gan_losses = []
    for i, data in enumerate(data_loader):
        # Train discriminator
        discriminator_loss = model.train_discriminator(data.to(device))
        # Train generator every n_critic steps
        if i % gan.n_critic == 0:
            generator_loss = model.train_generator(data.to(device))
            gan_losses.append([discriminator_loss, generator_loss])
    print(f"Epoch {epoch+1}, Discriminator Loss: {np.mean([l[0] for l in gan_losses]):.4f}, Generator Loss: {np.mean([l[1] for l in gan_losses]):.4f}")
    # Update memory
    with torch.no_grad():
        for data in data_loader:
            model.update_memory(data.to(device))
    # Evaluate on test set
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(labels, scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("ROC-AUC", auc)