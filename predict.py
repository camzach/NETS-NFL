import numpy as np
import pickle as pkl

import yaml
from argparse import Namespace

import torch

from torch_models import multiPred_Transformer
torch.set_printoptions(precision=3, sci_mode=False)

device = 'cuda:0'
model_file = 'save/trajectory_velocity/trajectory_vel_predictor_February01,2021,2051.pth'
trajectory_file = 'save/trajectory_velocity/trajectory_settings_February01,2021,2051.yaml'
transformer_file = 'save/trajectory_velocity/Transformer_settings_February01,2021,2051.yaml'

with open(trajectory_file, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    config = Namespace(**config)
    config_str = str(config).replace('Namespace', '')

with open(transformer_file, 'r') as stream:
    t_config = yaml.load(stream, Loader=yaml.FullLoader)
    t_config = Namespace(**t_config)
    config_str += str(t_config).replace('Namespace', '')

patience = t_config.patience
num_epochs = t_config.num_epochs
learning_rate = float(t_config.learning_rate)
batch_size = t_config.batch_size
hidden_dim = t_config.hidden_dim

train_perc = config.train_perc
val_perc = config.val_perc
train_inp = config.train_inp
pred_len = config.pred_len

model = multiPred_Transformer(pred_type='trajectory',
                              out_dim=2*pred_len,
                              hidden_dim=hidden_dim,
                              num_layers=t_config.num_layers,
                              nhead=t_config.nhead,
                              dropout=t_config.dropout,
                              use_LSTM=t_config.use_LSTM).to(device)
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint)

save_path = 'data/weeks/week1.pkl'
play_df = pkl.load(open(save_path, 'rb'))

inp_len = max(train_inp)

X = []
n_max = 1e6
if len(play_df) > n_max:
    play_df = play_df.sample(n=n_max, random_state=42)

target_shape = (play_df.iloc[0]['play'].shape[0], 46)

for it, p in play_df.iterrows():
    inp_vals = p['play']
    #inp_vals = flatten_features(inp_vals)
    
    if inp_vals.shape != target_shape:
        # print(f'wrong shape: {inp_vals.shape}')
        continue
    
    X.append(inp_vals)


play_seq = np.array(X, dtype='float')

# add sequences of variable input lengths, fixed output length
if pred_len + inp_len > play_seq.shape[1]:
    raise Exception('sequence not long enough!')

inp_seq = play_seq[:, :inp_len]
pred_seq = play_seq[:, inp_len:(inp_len+pred_len)]
temp = pred_seq.reshape( (pred_seq.shape[0], pred_seq.shape[1], -1, 2) )
pred_seq = np.transpose(temp, (0, 2, 1, 3)).reshape( (pred_seq.shape[0], 23, -1) )

N_train = int(play_seq.shape[0] * train_perc)
N_val = N_train + int(play_seq.shape[0] * val_perc)

stuff = torch.tensor([inp_seq[:5]], dtype=torch.float).to(device)

# while stuff.shape[0] < 100:
# a = stuff[-10:].reshape(5,10,46)
b = model(stuff)
print(b.shape)
# c = stuff[-1] + b
# print(c)
