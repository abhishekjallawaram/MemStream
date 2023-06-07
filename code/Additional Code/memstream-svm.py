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
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import OneClassSVM



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
parser.add_argument("--act_fn",help="Activation fucntion", default="Tanh")
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



from sklearn.preprocessing import MinMaxScaler

class MemStream(torch.nn.Module):
    def __init__(self, input_size, params):
        super(MemStream, self).__init__()
        self.memory_len = params['memory_len']
        self.nu = params['nu']
        self.kernel = params['kernel']
        self.gamma = params['gamma']
        self.svm = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.memory = torch.zeros(self.memory_len, input_size)
        self.memory_counter = 0

    def initialize_memory(self, data):
        n = len(data)
        self.memory[:n] = data
        self.memory_counter = n

    def train(self, data):
        n = len(data)
        if self.memory_counter + n <= self.memory_len:
            self.memory[self.memory_counter:self.memory_counter+n] = data
            self.memory_counter += n
        else:
            num_to_remove = self.memory_counter + n - self.memory_len
            self.memory[0:num_to_remove] = self.memory[-num_to_remove:]
            self.memory[-n:] = data
            self.memory_counter = self.memory_len
        self.svm.fit(self.memory)

    def forward(self, data):
        return self.svm.score_samples(data)



        
if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
    labels = np.loadtxt(lfile, delimiter=',')

if args.dataset == 'KDD':
    labels = 1 - labels
torch.manual_seed(args.seed)
N = args.memlen
params = {
          'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr , 'in_dim' : numeric[0].shape[0]
         }

if args.dataset == 'SYN':
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ',')).reshape(-1, 1)
    labels = np.loadtxt(lfile, delimiter=',')

print(params)
    
model = MemStream(numeric[0].shape[0], params={'memory_len': args.memlen, 'nu': 0.5, 'kernel': 'rbf', 'gamma': 'scale'}).to(device)

batch_size = 1
print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.initialize_memory(init_data[:args.memlen])
model.train(init_data)
err = []
for data in data_loader:
    output = model(data.to(device))
    err.append(output)
scores = np.array(err, dtype=np.float32)
auc = metrics.roc_auc_score(labels, scores)
print(f"Execution time: {time.time()-start_time:.2f} seconds")
print("ROC-AUC", auc)

if args.RQ1:
    from sklearn import metrics
    from tqdm import tqdm
    import os

    # Define hyperparameters to test
    nu_values = [0.1, 0.3, 0.5]
    kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma_values = ['scale', 'auto']
    memory_len_values = [256]

    file_num = 1
    output_file = args.dataset + '_RQ1_SVM.txt'
    while os.path.exists(output_file):
        output_file = args.dataset + '_RQ1_SVM' + str(file_num) + '.txt'
        file_num += 1

    best_auc = 0.0
    best_params = {}

    # Iterate over all combinations of hyperparameters
    with open(output_file, 'w') as f:
        f.write("Hyperparameter tuning results:\n")
        for nu in tqdm(nu_values):
            for kernel in kernel_values:
                for gamma in gamma_values:
                    for memory_len in memory_len_values:
                        params = {
                            'nu': nu,
                            'kernel': kernel,
                            'gamma': gamma,
                            'memory_len': memory_len
                        }
                        model = MemStream(numeric[0].shape[0], params)
                        batch_size = 1
                        #print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
                        data_loader = DataLoader(numeric, batch_size=batch_size)
                        init_data = numeric[labels == 0][:N].to(device)
                        model.initialize_memory(init_data[:memory_len])
                        model.train(init_data)
                        err = []
                        for data in data_loader:
                            output = model(data.to(device))
                            err.append(output)
                        score = np.array(err, dtype=np.float32)
                        auc = metrics.roc_auc_score(labels, scores)
                        avg_score = auc
                        f.write(f"Params: {params}, ROC-AUC score: {avg_score}\n")
                        print(f"Params: {params}, ROC-AUC score: {avg_score}")
                        if avg_score > best_auc:
                            best_auc = avg_score
                            best_params = params
        f.write(f"Best score: {best_auc}\n")
        f.write(f"Best params: {best_params}\n")
        print("Best score:", best_auc)
        print("Best params:", best_params)


