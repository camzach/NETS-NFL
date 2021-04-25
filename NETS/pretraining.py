import numpy as np
import yaml
from argparse import Namespace
from shutil import copyfile
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils.get_data import get_sequences
from torch_models import LSTM_Transformer
torch.set_printoptions(precision=3, sci_mode=False)

def flatten_features(inp):
    inp_l = inp.tolist()
    flat = []
    for a in inp_l:
        f = []
        for b in a:
            try:
                f = f + list(b)
            except:
                f.append(b)
        flat.append(f)
    return np.array(flat)

device = 'cuda:0'
dt = 0.12
def pretrain(yaml_pretraining, yaml_transformer):
    yaml_file = yaml_pretraining + '.yaml'
    with open(yaml_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        config = Namespace(**config)
        config_str = str(config).replace('Namespace', '')
    
    train_perc = config.train_perc
    train_inp = config.train_inp
    pred_len = config.pred_len
    save_folder = config.folder
    time_str = datetime.datetime.now().strftime("%B%d,%Y,%H%M")
    save_yaml_file = save_folder + '/' + yaml_pretraining + time_str + '.yaml'
    copyfile(yaml_file, save_yaml_file)
    
    inp_len = max(train_inp)
    
    print('extract trajectories', flush=True)
    play_df = get_sequences('possession_data', pred_len+inp_len, inp_len)
    
    # random shuffle and extract
    X = []
    play_df = play_df.sample(n=len(play_df), random_state=42)
    inp_shape = play_df.iloc[0].play.shape
    for it, p in play_df.iterrows():
        inp_vals = p['play']
        if inp_vals.shape != inp_shape:
            continue
        X.append(inp_vals)
    play_seq = np.array(X, dtype='float')
    
    inp_seq = play_seq[:, :inp_len]
    pred_seq = play_seq[:, inp_len:(inp_len+pred_len)]
    temp = pred_seq.reshape( (pred_seq.shape[0], pred_seq.shape[1], -1, 2) )
    pred_seq = np.transpose(temp, (0, 2, 1, 3)).reshape( (pred_seq.shape[0], 11, -1) )
    
    N_train = int(play_seq.shape[0] * train_perc)
    N_val = N_train + int(play_seq.shape[0] * (1-train_perc)/2)
    
    X_train = inp_seq[:N_train]
    y_train = pred_seq[:N_train]
    
    X_val = inp_seq[N_train:N_val]
    y_val = pred_seq[N_train:N_val]
    
    X_train = np.array(X_train, dtype='float')
    y_train = np.array(y_train, dtype='float')
    X_val = np.array(X_val, dtype='float')
    y_val = np.array(y_val, dtype='float')
    
    print(f'Training shape: {X_train.shape}')
    print(f'Validation shape: {X_val.shape}')
    
    yaml_file = yaml_transformer + '.yaml'
    with open(yaml_file, 'r') as stream:
        t_config = yaml.load(stream, Loader=yaml.FullLoader)
        t_config = Namespace(**t_config)
        config_str += str(t_config).replace('Namespace', '')
    save_yaml_file = save_folder + '/' + yaml_transformer + \
        time_str + '.yaml'
    copyfile(yaml_file, save_yaml_file)
    
    patience = t_config.patience
    num_epochs = t_config.num_epochs
    learning_rate = float(t_config.learning_rate)
    batch_size = t_config.batch_size
    hidden_dim = t_config.hidden_dim
    model = LSTM_Transformer(pred_type='trajectory',
                        out_dim=2*pred_len,
                        hidden_dim=hidden_dim,
                        n_offense=t_config.n_offense,
                        n_defense=t_config.n_defense,
                        num_layers=t_config.num_layers,
                        nhead=t_config.nhead,
                        dropout=t_config.dropout).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = 1e9; best_epoch = 0
    # Train the model
    for epoch in range(num_epochs):
        # X is a torch Variable
        n_train = X_train.shape[0]
        permutation = torch.randperm(n_train)
        
        loss_avg = 0
        model.train()
        for i in range(0, n_train, batch_size):
            indices = permutation[i:i+batch_size]
            
            for L in train_inp:
                batch_x, batch_y = X_train[indices], y_train[indices]
                batch_x = batch_x[:, -L:]
                
                batch_x = Variable(torch.from_numpy(batch_x)).to(device)
                batch_y = Variable(torch.from_numpy(batch_y)).to(device)
                n_batch = batch_x.shape[0]
                batch_y = batch_y.float()
                
                last_pos = batch_x[:, -1, :].reshape( (n_batch, -1, 2) ).float()
                gt_plus = torch.cat( [last_pos, batch_y], dim=-1 )
                gt_v = ( gt_plus[:,:,2:] - gt_plus[:,:,:-2] ) / dt
                
                outs = model(batch_x.float())
                
                # obtain the loss function
                loss = criterion(outs, gt_v)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if L == inp_len:
                    loss_avg += loss.item()*n_batch/n_train
        
        with torch.no_grad():
            model.eval()
            
            n_val = X_val.shape[0]
            idxs = torch.arange(n_val)
            val_loss_avg = 0
            loc_err = torch.zeros( (n_val, 11, pred_len) )
            for i in range(0, n_val, batch_size):
                indices = idxs[i:i+batch_size]
                batch_x, batch_y = X_val[indices], y_val[indices]
                batch_x = Variable(torch.from_numpy(batch_x)).to(device)
                batch_y = Variable(torch.from_numpy(batch_y)).to(device)     
                n_batch = batch_x.shape[0]
                batch_y = batch_y.float()
                
                outs = model(batch_x.float())
                
                last_pos = batch_x[:, -1, :].reshape( (n_batch, -1, 2) ).float()
                gt_plus = torch.cat( [last_pos, batch_y], dim=-1 )
                gt_v = ( gt_plus[:,:,2:] - gt_plus[:,:,:-2] ) / dt
                
                dx = outs[:,:,::2]; dy = outs[:,:,1::2]
                out_locs = torch.zeros(outs.shape).to(device)
                out_locs[:,:,::2] = last_pos[:, :, 0].unsqueeze(-1) + dx.cumsum(dim=-1)*dt
                out_locs[:,:,1::2] = last_pos[:, :, 1].unsqueeze(-1) + dy.cumsum(dim=-1)*dt
                
                # sqrt( dx^2+dy^2 )
                loc_err[i:i+batch_size] = \
                    torch.sqrt( (out_locs[:,:,::2] - batch_y[:,:,::2])**2 + 
                                (out_locs[:,:,1::2] - batch_y[:,:,1::2])**2 )
                
                loss = criterion(outs, gt_v)
                val_loss_avg += loss.item()*n_batch/n_val
            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                best_epoch = epoch
                
                torch.save(model.state_dict(), save_folder +\
                           '/'+config.name+time_str+'.pth')
                print('new best')
                b = [0]
                o = [1, 2, 3, 4, 5]
                d = [6, 7, 8, 9, 10]
                for typ, text in zip((b, o, d), ('ball', 'offense', 'defense')):
                    ADE = loc_err[:, typ].mean()
                    FDE = loc_err[:, typ, -1].mean()
                    print(f'{text}: ADE {ADE:.2f}, FDE {FDE:.2f}')
            if epoch - best_epoch > patience:
                break
            print(f'Epoch {epoch}: train loss {loss_avg:.3f}, val loss {val_loss_avg:.3f}')
            
    return time_str

