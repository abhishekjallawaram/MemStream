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
parser.add_argument("--act_fn",help="Activation fucntion", default="Tanh")
parser.add_argument("--RQ1", help="Hypertune best metrics : Default False", default=False)
parser.add_argument("--RQ2", help="Effect of Activation Functions : Default False",default=False)
parser.add_argument("--RQ3", help="Memory Poisoning Prevention Analysis",default=False)
parser.add_argument("--RQ4", help="Concept Drift",default=False)
parser.add_argument("--RQ5", help="Impact of Memory",default=False)
parser.add_argument("--sp",help="Sampling Method", default="rand")

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
        self.gamma = 0
        self.K = 3
        self.exp = torch.Tensor([self.gamma**i for i in range(self.K)]).to(device)
        
        hidden_dim1 = int((self.in_dim + self.out_dim) // 2)
        hidden_dim2 = int((hidden_dim1 + self.out_dim) // 2)
        
        
        self.encoder1 = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, self.out_dim),
            act_fn,
        ).to(device)

        self.encoder2 = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, self.out_dim),
            act_fn,
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, self.in_dim),
        ).to(device)

        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        #self.recon_loss_fn = nn.MSELoss()
        #self.enc_loss_fn = nn.MSELoss()
        self.recon_loss_fn = nn.BCEWithLogitsLoss()
        self.enc_loss_fn = nn.BCEWithLogitsLoss()
        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.count = 0

    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = new.to(device)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            encoded1 = self.encoder1(new + 0.001 * torch.randn_like(new).to(device))
            output = self.decoder(encoded1)
            recon_loss = self.recon_loss_fn(output, new)
            
            encoded2 = self.encoder2(output + 0.001 * torch.randn_like(new).to(device))
            enc_loss = self.enc_loss_fn(encoded2, encoded1.detach())
            
            loss = recon_loss + enc_loss
            #loss = (recon_loss + enc_loss) * 0.5
            
            # if loss.item() < 1e-5:
            #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            #     break
            
            loss.backward()
            self.optimizer.step()


    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            if least_used_pos < self.memory.size(0):
                self.memory[least_used_pos] = encoder_output
                self.mem_data[least_used_pos] = data
                self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
                self.count += 1
                return 1
        return 0


    def initialize_memory(self, x):
        mean, std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder1(new.to(device))
        self.memory.requires_grad = False
        self.mem_data = x.to(device)

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoded1 = self.encoder1(new + 0.001 * torch.randn_like(new).to(device))
        output = self.decoder(encoded1)
        encoded2 = self.encoder2(output + 0.001 * torch.randn_like(new).to(device))
        recon_loss = self.recon_loss_fn(output, new)
        enc_loss = self.enc_loss_fn(encoded2, encoded1.detach())
        loss = recon_loss + enc_loss
        self.update_memory(loss, encoded2, x.to(device))
        return loss
    
#     def forward(self, x):
#         new = (x - self.mean) / self.std
#         new[:, self.std == 0] = 0
#         encoder_output = self.encoder1(new)
#         output = self.decoder(encoder_output)
#         encoded2 = self.encoder2(output)
# #         loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
#         loss_values = (torch.topk(torch.norm(self.memory - encoded2, dim=1, p=1), k=self.K, largest=False)[0]*self.exp).sum()/self.exp.sum()
#         self.update_memory(loss_values, encoder_output, x.to(device))
#         return loss_values

class EnsembleMemStream():
    def __init__(self, num_models, params, act_fn):
        self.num_models = num_models
        self.models = []
        for i in range(num_models):
            self.models.append(MemStream(params['in_dim'],params=params, act_fn=act_fn))
            
    def train(self, numeric, labels, epochs):
        for i in range(self.num_models):
            print(f"Training Model {i+1}/{self.num_models}")
            np.random.seed(i)
            torch.manual_seed(i)
            N = params['memory_len']
            #K_NN = params['K']
            model = self.models[i]
            init_data = numeric[labels == 0][:N].to(device)
            model.mem_data = init_data
            torch.set_grad_enabled(True)
            model.train_autoencoder(Variable(init_data).to(device), epochs=epochs)
            torch.set_grad_enabled(False)
            model.initialize_memory(Variable(init_data[:N]))
            
            data_loader = DataLoader(numeric, batch_size=params['batch_size'])
            err = []
            for data in data_loader:
                output = model(data.to(device))
                err.append(output)
            scores = np.array([i.cpu() for i in err])
            auc = metrics.roc_auc_score(labels, scores)
            print(f"Model {i+1} ROC-AUC: {auc}")
            
    def predict(self, numeric):
        ensemble_scores = np.zeros(len(numeric))
        for model in self.models:
            data_loader = DataLoader(numeric, batch_size=params['batch_size'])
            err = []
            for data in data_loader:
                output = model(data.to(device))
                err.append(output)
            scores = np.array([i.cpu() for i in err])
            ensemble_scores += scores
        return ensemble_scores / self.num_models



        
if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
    labels = np.loadtxt(lfile, delimiter=',')
    act_fn = nn.Tanh()
# else:
#     act_fn = nn.LogSoftmax(dim=1)

    
act_fn = nn.Tanh()
    
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


# ensemble_model = EnsembleMemStream(num_models=20, params=params, act_fn=act_fn)

# print(f"Training Ensemble of {ensemble_model.num_models} models...")
# ensemble_model.train(numeric, labels, epochs=args.epochs)

# print("Making predictions using Ensemble model...")
# ensemble_scores = ensemble_model.predict(numeric)
# auc = metrics.roc_auc_score(labels, ensemble_scores)
# print(f"Execution time: {time.time()-start_time:.2f} seconds")
# print("Ensemble ROC-AUC:", auc)
    


if args.sp == 'rand':    

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
              'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
             }

    if args.dataset == 'SYN':
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ',')).reshape(-1, 1)
        labels = np.loadtxt(lfile, delimiter=',')

    model = MemStream(numeric[0].shape[0],params,act_fn).to(device)

    batch_size = params['batch_size']
    print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
    data_loader = DataLoader(numeric, batch_size=batch_size)
    init_data = numeric[labels == 0][:N].to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(labels, scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("ROC-AUC", auc)
    


    
elif args.sp =='str':
    if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter=','))
        labels = np.loadtxt(lfile, delimiter=',')
        act_fn = nn.Tanh()
    else:
        act_fn = nn.LogSoftmax(dim=1)

    if args.dataset == 'KDD':
        labels = 1 - labels
    torch.manual_seed(args.seed)
    N = args.memlen
    params = {
        'beta': args.beta, 'memory_len': N, 'batch_size': 1, 'lr': args.lr
    }

    if args.dataset == 'SYN':
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter=',')).reshape(-1, 1)
        labels = np.loadtxt(lfile, delimiter=',')

    # Split the data into training and testing sets
    # Stratified sampling is performed on the training set
    X_train, X_test, y_train, y_test = train_test_split(numeric, labels, test_size=0.2, stratify=labels, random_state=args.seed)

    model = MemStream(X_train[0].shape[0], params, act_fn).to(device)

    batch_size = params['batch_size']
    print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)

    # Use only the training set to initialize memory and train the model
    init_data = X_train[y_train == 0][:N].to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))

    # Use the testing set for evaluation
    data_loader = DataLoader(X_test, batch_size=batch_size)
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(y_test, scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("ROC-AUC", auc)

elif args.sp == 'ovr':
    if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
        labels = np.loadtxt(lfile, delimiter=',')
        act_fn = nn.Tanh()
    else:
        act_fn = nn.LogSoftmax(dim=1)

    if args.dataset == 'KDD':
        labels = 1 - labels

    # Oversample the minority class
    ros = RandomOverSampler(random_state=args.seed)
    numeric_resampled, labels_resampled = ros.fit_resample(numeric, labels)

    torch.manual_seed(args.seed)
    N = args.memlen
    params = {
              'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
             }

    if args.dataset == 'SYN':
        numeric_resampled = torch.FloatTensor(numeric_resampled.reshape(-1, 1))

    model = MemStream(numeric_resampled[0].shape[0],params,act_fn).to(device)

    batch_size = params['batch_size']
    print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
    data_loader = DataLoader(numeric_resampled, batch_size=batch_size)
    init_data = torch.FloatTensor(numeric_resampled[labels_resampled == 0][:N]).to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(labels_resampled, scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("ROC-AUC", auc)

elif args.sp == 'sm':
    if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
        numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ','))
        labels = np.loadtxt(lfile, delimiter=',')
        act_fn = nn.Tanh()
    else:
        act_fn = nn.LogSoftmax(dim=1)

    if args.dataset == 'KDD':
        labels = 1 - labels

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=args.seed)
    numeric_resampled, labels_resampled = smote.fit_resample(numeric, labels)

    torch.manual_seed(args.seed)
    N = args.memlen
    params = {
              'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
             }

    if args.dataset == 'SYN':
        numeric_resampled = torch.FloatTensor(numeric_resampled.reshape(-1, 1))

    model = MemStream(numeric_resampled[0].shape[0],params,act_fn).to(device)

    batch_size = params['batch_size']
    print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
    data_loader = DataLoader(numeric_resampled, batch_size=batch_size)
    init_data = torch.FloatTensor(numeric_resampled[labels_resampled == 0][:N]).to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])
    auc = metrics.roc_auc_score(labels_resampled, scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("ROC-AUC", auc)

elif args.sp == 'ens_m':

    ensemble_model = EnsembleMemStream(num_models=5, params=params, act_fn=act_fn)

    print(f"Training Ensemble of {ensemble_model.num_models} models...")
    ensemble_model.train(numeric, labels, epochs=args.epochs)

    print("Making predictions using Ensemble model...")
    ensemble_scores = ensemble_model.predict(numeric)
    auc = metrics.roc_auc_score(labels, ensemble_scores)
    print(f"Execution time: {time.time()-start_time:.2f} seconds")
    print("Ensemble ROC-AUC:", auc)

#RQ1 hyper-parametertuning
if args.RQ1:
    learning_rates = [1,1e-1,1e-2, 1e-3,1e-4]
    num_epochs = 5000
    if len(numeric) < 2000:
        memory_sizes = [4, 8, 16, 32, 64]
    else:
        #memory_sizes = [128, 256, 512, 1024, 2048]
        memory_sizes = [128,256,512,1024,2048]
    thresholds = [1, 1e-1,1e-2,1e-3,1e-4,1e-5]

    import time
    from tqdm import tqdm
    import os
    
    file_num = 1
    output_file = args.dataset+'_ede_RQ1.txt'
    while os.path.exists(output_file):
        output_file = args.dataset+'_ede_RQ1_'+str(file_num)+'.txt'
        file_num += 1
    
    output_file = open(output_file, 'w')

    best_auc = 0.0
    best_params = {}

    
    output_file.write(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'AUC':<15} {'Time':<15}\n")
    print(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'AUC':<15} {'Time':<15}")

    for lr in tqdm(learning_rates,desc="Learning Rates"):
        #for mem_size in tqdm(memory_sizes, desc="Memory Size"):
        for mem_size in memory_sizes:
            #for thres in tqdm(thresholds, desc="Beta"):
            for thres in thresholds:
                start_time_1 = time.time()
                params = {
                    'memory_len': mem_size,
                    'beta': thres,
                    'lr': lr,
                    'epochs': num_epochs,
                    'batch_size': batch_size
                }
                model = MemStream(numeric[0].shape[0], params,act_fn).to(device)

                init_data = numeric[labels == 0][:N].to(device)
                model.mem_data = init_data

                torch.set_grad_enabled(True)
                model.train_autoencoder(Variable(init_data).to(device), epochs=num_epochs)
                torch.set_grad_enabled(False)
                model.initialize_memory(Variable(init_data[:N]))

                err = []
                for data in data_loader:
                    output = model(data.to(device))
                    err.append(output)
                scores = np.array([i.cpu() for i in err])
                auc = metrics.roc_auc_score(labels, scores)

                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        'memory_len': mem_size,
                        'beta': thres,
                        'lr': lr,
                        'epochs': num_epochs,
                        'batch_size': batch_size
                    }

               
                output_file.write(f"{lr:<15} {mem_size:<15} {thres:<15} {auc:.4f} {time.time()-start_time_1:.2f}s\n")
                print(f"{lr:<15} {mem_size:<15} {thres:<15} {auc:.4f} {time.time()-start_time_1:.2f}s")

    output_file.write("\nBest Parameters: "+str(best_params)+"\n")
    output_file.write("Best ROC-AUC: "+str(best_auc)+"\n")
    
    # print("\nBest Parameters:", best_params)
    # print("Best ROC-AUC:", best_auc)


    print("Best Parameters:", best_params)
    print("Best ROC-AUC:", best_auc)
    
    model_name = args.dataset+'_RQ1_'+str(file_num) + '.pt'
    torch.save(model.state_dict(), model_name)
    output_file.write("Best Model saved as: "+str(model_name)+"\n")
    
#impact of activation function
if args.RQ2:
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    params = {
        'beta': args.beta, 
        'memory_len': N, 
        'batch_size': 1, 
        'lr': args.lr
    }
    
    activation_functions = {
        'Softsign': nn.Softsign(), 
        'LogSoftmax': nn.LogSoftmax(dim=1), 
        'Tanh': nn.Tanh(), 
        'Softmax': nn.Softmax(dim=1),
        'Softmin': nn.Softmin(dim=1)
    }
    
    num_epochs = 5000
    
    file_num = 1
    output_file = args.dataset+'_RQ2.txt'
    while os.path.exists(output_file):
        output_file = args.dataset+'_RQ2_'+str(file_num)+'.txt'
        file_num += 1
    
    output_file = open(output_file, 'w')
    
    best_model = None
    best_auc = 0
    output_file.write(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}\n")
    print(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}")
    file_num = 1
    output_file1 = args.dataset+'_RQ2.pdf'
    while os.path.exists(output_file1):
        output_file1 = args.dataset+'_RQ2_'+str(file_num)+'.pdf'
        file_num += 1
    
    figs = []
    auc_dict = {}
    with PdfPages(output_file1) as pdf:
        for name, act_fn in activation_functions.items():
            start_time_1 = time.time()
            model = MemStream(numeric[0].shape[0], params, act_fn).to(device)
            init_data = numeric[labels == 0][:N].to(device)
            model.mem_data = init_data

            torch.set_grad_enabled(True)
            model.train_autoencoder(Variable(init_data).to(device), epochs=num_epochs)
            torch.set_grad_enabled(False)
            model.initialize_memory(Variable(init_data[:N]))

            err = []
            for data in data_loader:
                output = model(data.to(device))
                err.append(output)
            scores = np.array([i.cpu() for i in err])
            auc = metrics.roc_auc_score(labels, scores)

            if auc > best_auc:
                best_auc = auc
                best_params = {
                    'memory_len': args.memlen,
                    'beta': args.beta,
                    'lr': args.lr,
                    'epochs': num_epochs,
                    'activation_fn': name
                }
        
            auc_dict[name] = scores
        
            output_file.write(f"{args.lr:<15} {N:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s\n")
            print(f"{args.lr:<15} {N:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s")
        
            fig = plt.figure()
            plt.title(f"Activation Function: {name}")
            plt.xlabel("Epochs")
            plt.ylabel("ROC-AUC")
            plt.ylim(0.5, 1.0)
            plt.grid(True)
            plt.plot(range(len(scores)), scores)
            pdf.savefig(fig)
            figs.append(fig)
            plt.close(fig)
    
        fig = plt.figure()
        plt.title(f"ROC-AUC vs Activation Function")
        plt.xlabel("Activation Function")
        plt.ylabel("ROC-AUC")
        plt.ylim(0.0, 1.0)
        plt.grid(True)
        plt.bar(auc_dict.keys(), [metrics.roc_auc_score(labels, auc_dict[name]) for name in auc_dict.keys()])
        pdf.savefig(fig)
        figs.append(fig)
        plt.close(fig)

    print(f"Best ROC-AUC: {best_auc:.4f}")
    print(f"Best params: {best_params}")
    output_file.write("\nBest Parameters: "+str(best_params)+"\n")
    output_file.write("Best ROC-AUC: "+str(best_auc)+"\n")
    output_file.close()

# #impact of activation function
# if args.RQ2:
#     params = {
#           'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
#          }
#     activation_functions = {'Softsign': nn.Softsign(), 
#                             'LogSoftmax': nn.LogSoftmax(dim=1), 
#                             'Tanh': nn.Tanh(), 
#                             'Softmax': nn.Softmax(dim=1),
#                             'Softmin': nn.Softmin(dim=1)
#                            }
#     num_epochs = 5000
#     import time
#     from tqdm import tqdm
#     import os
    
#     file_num = 1
#     output_file = args.dataset+'_RQ2.txt'
#     while os.path.exists(output_file):
#         output_file = args.dataset+'_RQ2_'+str(file_num)+'.txt'
#         file_num += 1
    
#     output_file = open(output_file, 'w')
    
#     best_model = None
#     best_auc = 0
#     output_file.write(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}\n")
#     print(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}")
#     for name, act_fn in activation_functions.items():
#         start_time_1 = time.time()
#         model = MemStream(numeric[0].shape[0], params, act_fn).to(device)
#         init_data = numeric[labels == 0][:N].to(device)
#         model.mem_data = init_data

#         torch.set_grad_enabled(True)
#         model.train_autoencoder(Variable(init_data).to(device), epochs=num_epochs)
#         torch.set_grad_enabled(False)
#         model.initialize_memory(Variable(init_data[:N]))

#         err = []
#         for data in data_loader:
#             output = model(data.to(device))
#             err.append(output)
#         scores = np.array([i.cpu() for i in err])
#         auc = metrics.roc_auc_score(labels, scores)

#         if auc > best_auc:
#             best_auc = auc
#             best_params = {
#                 'memory_len': args.memlen,
#                 'beta': args.beta,
#                 'lr': args.lr,
#                 'epochs': num_epochs,
#                 'activation_fn': name
#             }

               
#         output_file.write(f"{args.lr:<15} {N:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s\n")
#         print(f"{args.lr:<15} {N:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s")

#     output_file.write("\nBest Parameters: "+str(best_params)+"\n")
#     output_file.write("Best ROC-AUC: "+str(best_auc)+"\n")
    
#     print("Best Parameters:", best_params)
#     print("Best ROC-AUC:", best_auc)

#Memory Poisoning Prevention Analysis
if args.RQ3:
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    file_num = 1
    output_file = args.dataset+'_RQ3.txt'
    while os.path.exists(output_file):
        output_file = args.dataset+'_RQ3_'+str(file_num)+'.txt'
        file_num += 1
    
    output_file = open(output_file, 'w')
    params = {'memory_len': N, 'batch_size': 1, 'lr': 1e-2}
    best_auc = 0.0
    best_params = {}
    betas = [1, 0.1, 0.01, 0.001, 0.0001]
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    output_file1 = args.dataset+'_RQ3.pdf'
    while os.path.exists(output_file1):
        output_file1 = args.dataset+'_RQ3_'+str(file_num)+'.pdf'
        file_num += 1

    figs = []
    with PdfPages(args.dataset+'_RQ3_plots.pdf') as pdf:
        for beta in betas:
            fig = plt.figure()
            plt.title(f"Beta = {beta}")
            plt.xlabel("Gamma")
            plt.ylabel("ROC-AUC")
            plt.ylim(0.5, 1.0)
            plt.grid(True)
            auc_vals = []
            for gamma in gammas:
                params['beta'] = beta
                params['gamma'] = gamma
                print(f"Testing beta={beta}, gamma={gamma}")
                start_time = time.time()
                model = MemStream(numeric[0].shape[0], params,act_fn).to(device)
                data_loader = DataLoader(numeric, batch_size=params['batch_size'])
                init_data = numeric[labels == 0][:N].to(device)
                model.mem_data = torch.cat((init_data, numeric[labels == 1][:1].to(device)), dim=0)
                torch.set_grad_enabled(True)
                model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
                torch.set_grad_enabled(False)
                model.initialize_memory(Variable(model.mem_data[:N]))
                err = []
                for data in data_loader:
                    output = model(data.to(device))
                    err.append(output)
                scores = np.array([i.cpu() for i in err])
                auc = metrics.roc_auc_score(labels, scores)
                print(f"ROC-AUC: {auc:.4f}")
                if auc > best_auc:
                    best_auc = auc
                    best_params = {'beta': beta, 'gamma': gamma}
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f}s")
                auc_vals.append(auc)
            plt.plot(gammas, auc_vals)
            pdf.savefig(fig)
            figs.append(fig)
            plt.close(fig)
            
        print(f"Best ROC-AUC: {best_auc:.4f}")
        print(f"Best params: {best_params}")
        output_file.write("\nBest Parameters: "+str(best_params)+"\n")
        output_file.write("Best ROC-AUC: "+str(best_auc)+"\n")
        
if args.RQ4:
    with open('conceptdriftdata.txt', 'r') as f:
        lines = f.readlines()

    values = [float(line.strip()) for line in lines]
    indices = range(len(values))

    plt.plot(indices, values)
    plt.xlabel('data_index')
    plt.ylabel('value')
    plt.savefig('concept_drift.pdf', format='pdf')
    plt.clf()
    
    from torch.utils.data import DataLoader, TensorDataset
    
    numeric = torch.FloatTensor(np.loadtxt('conceptdriftdata.txt', delimiter = ',')).reshape(-1, 1)
    data_loader = DataLoader(numeric, batch_size=batch_size)
    init_data = numeric[:N].to(device)
    model.mem_data = init_data
    torch.set_grad_enabled(True)
    model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
    torch.set_grad_enabled(False)
    model.initialize_memory(Variable(init_data[:N]))
    err = []
    for data in data_loader:
        output = model(data.to(device))
        err.append(output)
    scores = np.array([i.cpu() for i in err])

    plt.plot(range(len(scores)), scores)
    plt.xlabel('Iteration')
    plt.ylabel('Anomaly Score')
    plt.savefig('concept_drift_score.pdf', format='pdf')
    plt.clf()

#Impact of Memory
if args.RQ5:
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    n = 14
    memory_len = [pow(2, i) for i in range(2, n+1)]
    num_epochs = 5000
    
    file_num = 1
    output_file = args.dataset+'_RQ5.txt'
    while os.path.exists(output_file):
        output_file = args.dataset+'_RQ5_'+str(file_num)+'.txt'
        file_num += 1
    
    output_file = open(output_file, 'w')
    
    best_model = None
    best_auc = 0
    output_file.write(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}\n")
    print(f"{'Learning Rate':<15} {'Memory Size':<15} {'Beta':<15} {'Epochs':<15} {'Act_fn':<15} {'AUC':<15} {'Time':<15}")
    file_num = 1
    output_file1 = args.dataset+'_RQ5.pdf'
    while os.path.exists(output_file1):
        output_file1 = args.dataset+'_RQ5_'+str(file_num)+'.pdf'
        file_num += 1
    
    figs = []
    auc_dict = {}
    with PdfPages(output_file1) as pdf:
        for mem in memory_len:
            params = {
                'beta': args.beta, 
                'act_fn' : nn.Tanh(),
                'batch_size': 1, 
                'lr': args.lr,
                'memory_len': mem
                }
            start_time_1 = time.time()
            model = MemStream(numeric[0].shape[0], params, act_fn).to(device)
            init_data = numeric[labels == 0][:mem].to(device) # use mem instead of N
            model.mem_data = init_data

            torch.set_grad_enabled(True)
            model.train_autoencoder(Variable(init_data).to(device), epochs=num_epochs)
            torch.set_grad_enabled(False)
            model.initialize_memory(Variable(init_data))

            err = []
            for data in data_loader:
                output = model(data.to(device))
                err.append(output)
            scores = np.array([i.cpu() for i in err])
            auc = metrics.roc_auc_score(labels, scores)

            if auc > best_auc:
                best_auc = auc
                best_params = {
                    'memory_len': mem, 
                    'beta': args.beta,
                    'lr': args.lr,
                    'epochs': num_epochs,
                    'activation_fn': "Tanh"
                }
            name = "Tanh"
        
            auc_dict[mem] = auc 
        
            output_file.write(f"{args.lr:<15} {mem:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s\n")
            print(f"{args.lr:<15} {mem:<15} {args.beta:<15} {name:<15} {num_epochs:<15} {auc:.4f} {time.time()-start_time_1:.2f}s")
        
    
        fig = plt.figure()
        plt.title("ROC-AUC vs Memory Size")
        plt.xlabel("Memory Size")
        plt.ylabel("ROC-AUC")
        plt.plot(list(auc_dict.keys()), list(auc_dict.values()))
        pdf.savefig(fig)
        plt.close()
