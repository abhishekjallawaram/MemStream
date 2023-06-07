import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.spatial as sp
import argparse
import scipy.io

from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

start_time = time.time()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# tf.test.gpu_device_name()
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NSL')
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument("--dev", help="device", default="GPU")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=2048)
parser.add_argument("--seed", type=int, help="random seed", default=0)
args = parser.parse_args()

np.random.seed(args.seed)

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
    df = scipy.io.loadmat(args.dataset+".mat")
    numeric = tf.convert_to_tensor(df['X'], dtype=tf.float32)
    labels = (df['y']).astype(float).reshape(-1)


device = args.dev


class MemStream(Model):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = in_dim*2
        self.memory_len = params['memory_len']
        self.max_thres = K.constant(params['beta'], dtype='float32')
        self.memory = K.random_normal(shape=(self.memory_len, self.out_dim), dtype='float32')
        self.mem_data = K.random_normal(shape=(self.memory_len, self.in_dim), dtype='float32')
        self.mem_data._trainable = False
        self.memory._trainable = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.encoder = Sequential([
            layers.Dense(self.out_dim, activation='log_softmax', input_shape=(self.in_dim,))
        ])  
        self.decoder = Sequential([
            layers.Dense(self.in_dim)
        ])
        self.clock = 0
        self.last_update = -1
        self.optimizer = Adam(lr=params['lr'])
        self.loss_fn = MeanSquaredError()
        self.count = 0


    def train_autoencoder(self, data, epochs):
        self.mem_data = data
        self.mean, self.std = np.mean(self.mem_data, axis=0), np.std(self.mem_data, axis=0)
        new = (data - self.mean) / (self.std + 1e-7)
        new[:, self.std == 0] = 0
    
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                encoded = self.encoder(new + 0.001 * tf.random.normal(new.shape))
                decoded = self.decoder(encoded)
                loss = self.loss_fn(decoded, new)
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
            #print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.numpy():.4f}")
    

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = self.count % self.memory_len
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.count += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = np.mean(x, axis=0), np.std(x, axis=0)
        new_np = np.array(x)
        new_np[:, std == 0] = 0
        self.memory = self.encoder(new_np)
        self.memory.trainable = False
        self.mem_data = x


    def call(self, x):
        new = (x - self.mean) / self.std
        new = tf.where(self.std == 0, tf.zeros_like(x), new)
        encoder_output = self.encoder(new)
        loss_values = tf.norm(self.memory - encoder_output, axis=1, ord=1).numpy().min()
        self.update_memory(loss_values, encoder_output, x)
        return loss_values


if args.dataset in ['KDD', 'NSL', 'UNSW', 'DOS']:
    numeric = np.loadtxt(nfile, delimiter = ',')
    labels = np.loadtxt(lfile, delimiter=',')

if args.dataset == 'KDD':
    labels = 1 - labels
np.random.seed(args.seed)
N = args.memlen
params = {
          'beta': args.beta, 'memory_len': N, 'batch_size':1, 'lr':args.lr
         }

model = MemStream(numeric[0].shape[0], params)

batch_size = params['batch_size']
print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
data_loader = tf.data.Dataset.from_tensor_slices(numeric).batch(batch_size)

init_data = tf.boolean_mask(numeric, labels == 0)[:N]
model.mem_data = init_data
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
model.train_autoencoder(init_data.numpy(), args.epochs)
# model.memory.requires_grad = False
# model.mem_data.requires_grad = False
model.initialize_memory(init_data)




err = []

for data in data_loader:
    output = model.predict(data)
    err.append(output)

scores = np.array([i for i in err])
auc = metrics.roc_auc_score(labels, scores)

print(f"Execution time: {time.time()-start_time:.2f} seconds")
print("ROC-AUC", auc)

