import numpy as np
import pandas as pd
import yaml
from argparse import Namespace
from shutil import copyfile
import datetime
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch_models import LSTM_Transformer
torch.set_printoptions(precision=3, sci_mode=False)


def get_NFL_sequences(path):
    data = pkl.load(open(path, 'rb'))
    return data

def init_model(yaml_transformer, save_folder, time_str, pred_len):
    yaml_file = yaml_transformer + '.yaml'
    with open(yaml_file, 'r') as stream:
        t_config = yaml.load(stream, Loader=yaml.FullLoader)
        t_config = Namespace(**t_config)
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
    return model, patience, num_epochs, learning_rate, batch_size
    
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
    
    train_perc = config.train_perc
    train_inp = config.train_inp
    pred_len = config.pred_len
    save_folder = config.folder
    time_str = datetime.datetime.now().strftime("%B%d,%Y,%H%M")
    save_yaml_file = save_folder + '/' + yaml_pretraining + time_str + '.yaml'
    copyfile(yaml_file, save_yaml_file)
    
    inp_len = max(train_inp)
    
    print('extract trajectories', flush=True)
    play_df = get_NFL_sequences('../data/sequences.pkl')
    
    # random shuffle and extract
    inp_seq = []; pred_seq = []
    for it, p in play_df.iterrows():
        past = np.nan_to_num(p.past, nan=-1)
        future = np.nan_to_num(p.future, nan=-1)
        inp_seq.append(past)
        pred_seq.append(future)
    
    N = len(play_df)
    n_players = play_df.iloc[0].past.shape[1]
    inp_seq = np.array(inp_seq).reshape( (N, inp_len, -1) )
    pred_seq = np.transpose( np.array(pred_seq), (0,2,1,3)).reshape( (N, n_players, -1) )
    
    N_train = int(N * train_perc)
    N_val = N_train + int(N * (1-train_perc)/2)
    
    X_train = np.array(inp_seq[:N_train], dtype='float')
    y_train = np.array(pred_seq[:N_train], dtype='float')
    X_val = np.array(inp_seq[N_train:N_val], dtype='float')
    y_val = np.array(pred_seq[N_train:N_val], dtype='float')
    X_test = np.array(inp_seq[N_val:], dtype='float')
    y_test = np.array(pred_seq[N_val:], dtype='float')
    
    print(f'Training input shape: {X_train.shape}')
    print(f'Training prediction shape: {y_train.shape}')
    
    model, patience, num_epochs, learning_rate, batch_size =\
        init_model(yaml_transformer, save_folder, time_str, pred_len)
    
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
                gt_plus = torch.cat( [last_pos, batch_y], dim=2 )
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
            loc_err = torch.zeros( (n_val, n_players, pred_len) )
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
    return time_str, X_test, y_test, play_df

def testing(yaml_pretraining, yaml_transformer, load_date, X_test, y_test, play_df):
    transformer_config = yaml_pretraining + '.yaml'
    with open(transformer_config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        config = Namespace(**config)
    save_folder = config.folder
    pred_len = config.pred_len
    
    trajectory_config = save_folder + '/' + yaml_transformer + load_date + '.yaml'
    with open(trajectory_config, 'r') as stream:
        t_config = yaml.load(stream, Loader=yaml.FullLoader)
        t_config = Namespace(**t_config)
    
    model = LSTM_Transformer(pred_type='trajectory',
                            out_dim=2*pred_len,
                            n_offense=t_config.n_offense,
                            n_defense=t_config.n_defense,
                            hidden_dim=t_config.hidden_dim,
                            num_layers=t_config.num_layers,
                            nhead=t_config.nhead,
                            dropout=t_config.dropout).to(device)
    load_model_path = save_folder + '/' + config.name + load_date + '.pth'
    model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    
    batch_size = 128
    # testing
    with torch.no_grad():
        model.eval()
        
        n_test = X_test.shape[0]
        idxs = torch.arange(n_test)
        predictions = []; past = []; future = []
        for i in range(0, n_test, batch_size):
            indices = idxs[i:i+batch_size]
            batch_x, batch_y = X_test[indices], y_test[indices]
            batch_x = Variable(torch.from_numpy(batch_x)).to(device)
            batch_y = Variable(torch.from_numpy(batch_y)).to(device)     
            n_batch = batch_x.shape[0]
            batch_y = batch_y.float()
            
            outs = model(batch_x.float())
            
            last_pos = batch_x[:, -1, :].reshape( (n_batch, -1, 2) ).float()
            dx = outs[:,:,::2]; dy = outs[:,:,1::2]
            out_locs = torch.zeros(outs.shape).to(device)
            out_locs[:,:,::2] = last_pos[:, :, 0].unsqueeze(-1) + dx.cumsum(dim=-1)*dt
            out_locs[:,:,1::2] = last_pos[:, :, 1].unsqueeze(-1) + dy.cumsum(dim=-1)*dt
            
            predictions.append(out_locs.detach().cpu().numpy())
            past.append(X_test[indices])
            future.append(y_test[indices])
    
    predictions = np.concatenate(predictions, axis=0)
    n_players = predictions.shape[1]
    predictions = predictions.reshape((n_test, n_players, pred_len, 2))
    predictions = np.transpose(predictions, (0,2,1,3))
    past = np.concatenate(past, axis=0).reshape( (n_players, n_test, -1, 2) )
    past = np.transpose(past, (1,0,2,3))
    future = np.concatenate(future, axis=0)
    future = future.reshape((n_test, n_players, pred_len, 2))
    future = np.transpose(future, (0,2,1,3))
    
    
    test_df = pd.DataFrame({'past': [past[i] for i in range(n_test)],
                            'future': [future[i] for i in range(n_test)],
                            'predictions': [predictions[i] for i in range(n_test)]})
    test_df.to_pickle('saved_data/NFLsequences_with_predictions_2.pkl')

if __name__ == '__main__':    
    trajectory_yaml = 'NFL_trajectory_settings'
    transformer_yaml = 'NFL_transformer_settings'
    time_str, X_test, y_test, play_df = pretrain(trajectory_yaml, transformer_yaml)
    
    testing(trajectory_yaml, transformer_yaml, time_str, X_test, y_test, play_df)
