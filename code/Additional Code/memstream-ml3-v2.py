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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import os
 

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
parser.add_argument("--fs",help="Sampling Method", default="ac")
parser.add_argument("--RQ1", help="Hypertune best metrics : Default False", default=False)
parser.add_argument("--RQ2", help="Effect of Activation Functions : Default False",default=False)
parser.add_argument("--RQ3", help="Memory Poisoning Prevention Analysis",default=False)
parser.add_argument("--RQ4", help="Concept Drift",default=False)
parser.add_argument("--RQ5", help="Impact of Memory",default=False)

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
        
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            act_fn,
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.in_dim)
        ).to(device)
        
        self.clock = 0
        self.last_update = -1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['lr'])
        self.loss_fn = nn.MSELoss()
        self.count = 0


    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (data - self.mean) / self.std
        new[:, self.std == 0] = 0
        new = new.to(device)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.decoder(self.encoder(new + 0.001 * torch.randn_like(new).to(device)))
            loss = self.loss_fn(output, new)
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
        self.memory = self.encoder(new.to(device))
        self.memory.requires_grad = False
        self.mem_data = x.to(device)

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new.to(device))
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        #loss_values = torch.norm(self.memory - encoder_output, dim=1, p=2).min()
        #loss_values = (1 - F.cosine_similarity(self.memory, encoder_output, dim=1)).min()

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
    'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
         }

if args.dataset == 'SYN':
    numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter = ',')).reshape(-1, 1)
    labels = np.loadtxt(lfile, delimiter=',') 
    
if args.fs == 'pca_sd':
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    pca = PCA(n_components=0.95)
    numeric = pca.fit_transform(numeric_scaled)
elif args.fs == 'pca':
    pca = PCA(n_components=0.95)
    numeric = pca.fit_transform(numeric)
elif args.fs == 'ica_sd':
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    ica = FastICA(random_state=args.seed,whiten='unit-variance')
    numeric = ica.fit_transform(numeric_scaled)
elif args.fs == 'ica':
    ica = FastICA(random_state=args.seed,whiten='unit-variance')
    numeric = ica.fit_transform(numeric)
elif args.fs == 'lda_sd':
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    lda = LDA()
    numeric = lda.fit_transform(numeric_scaled, labels)
elif args.fs == 'lda':
    lda = LDA()
    numeric = lda.fit_transform(numeric, labels)
elif args.fs == 'chi':
    numeric = SelectKBest(chi2).fit_transform(numeric, labels)
elif args.fs == 'mi':
    selector = SelectKBest(mutual_info_classif)
    numeric_selected = selector.fit_transform(numeric, labels)
elif args.fs == 'rfe':
    clf = RandomForestClassifier()
    rfe = RFE(clf)
    numeric = rfe.fit_transform(numeric, labels)

numeric = torch.FloatTensor(numeric)
    
if args.sp == 'rand':    
        
    X_train, X_test, y_train, y_test = train_test_split(numeric, labels, test_size=0.2, random_state=args.seed)

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
    fpr, tpr, _ = metrics.roc_curve(y_test, scores)
    
    output_file = args.dataset+'_roc_rand.png'
    while os.path.exists(output_file):
        output_file = args.dataset+'_roc_rand'+str(file_num)+'.png'
        file_num += 1

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_file)
    plt.show()
    


    
elif args.sp =='str':
    
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
    output_file = args.dataset+'_roc_str.png'
    fpr, tpr, _ = metrics.roc_curve(y_test, scores)
    while os.path.exists(output_file):
        output_file = args.dataset+'_roc_str'+str(file_num)+'.png'
        file_num += 1

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_file)
    plt.show()

elif args.sp == 'ovr':

    # Oversample the minority class
    ros = RandomOverSampler(random_state=args.seed)
    numeric_resampled, labels_resampled = ros.fit_resample(numeric, labels)

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
    output_file = args.dataset+'_roc_ovr.png'
    fpr, tpr, _ = metrics.roc_curve(labels_resampled, scores)
    while os.path.exists(output_file):
        output_file = args.dataset+'_roc_ovr'+str(file_num)+'.png'
        file_num += 1

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_file)
    plt.show()


elif args.sp == 'sm':

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=args.seed)
    numeric_resampled, labels_resampled = smote.fit_resample(numeric, labels)

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
    output_file = args.dataset+'_roc_sm.png'
    fpr, tpr, _ = metrics.roc_curve(labels_resampled, scores)
    while os.path.exists(output_file):
        output_file = args.dataset+'_roc_sm'+str(file_num)+'.png'
        file_num += 1

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_file)
    plt.show()


#RQ1 hyper-parametertuning
if args.RQ1:
    learning_rates = [1, 1e-1, 1e-2, 1e-3]
    num_epochs = 5000
    if len(numeric) < 2000:
        memory_sizes = [4, 8, 16, 32, 64]
    else:
        memory_sizes = [128, 256, 512, 1024, 2048]
    thresholds = [10, 1, 0.1, 0.01, 0.001, 0.0001]

    import time
    from tqdm import tqdm
    import os
    
    file_num = 1
    output_file = args.dataset+'_RQ1.txt'
    while os.path.exists(output_file):
        output_file = args.dataset+'_RQ1_'+str(file_num)+'.txt'
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
    while os.path.exists(output_file):
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
