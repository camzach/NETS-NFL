import torch
import torch.nn as nn
from torch.autograd import Variable
import yaml
from argparse import Namespace

from torch_models import LSTM_Transformer
from data_utils.get_data import get_classifier_data

device = 'cuda'

def finetune(yaml_transformer, data_file, save_folder, new_name, 
              load_date=None, yaml_pretraining=None):
    X_train, y_train, X_val, y_val, X_test, y_test, n_classes, encoder =\
        get_classifier_data(data_file)
    
    print(f'Training shape: {X_train.shape}')
    print(f'Validation shape: {X_val.shape}')
    print(f'Test shape: {X_test.shape}')  

    print(y_train.shape)  

    transformer_config = yaml_transformer + '.yaml'
    with open(transformer_config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        config = Namespace(**config)
    if not load_date is None:
        
        trajectory_config = save_folder + '/' + yaml_pretraining + load_date + '.yaml'
        with open(trajectory_config, 'r') as stream:
            t_config = yaml.load(stream, Loader=yaml.FullLoader)
            t_config = Namespace(**t_config)
        pred_len = t_config.pred_len
        
        load_model = LSTM_Transformer(pred_type='trajectory',
                                        out_dim=2*pred_len,
                                        n_offense=config.n_offense,
                                        n_defense=config.n_defense,
                                        hidden_dim=config.hidden_dim,
                                        num_layers=config.num_layers,
                                        nhead=config.nhead,
                                        dropout=config.dropout).to(device)
        load_model_path = save_folder + '/' + t_config.name + load_date + '.pth'
        load_model.load_state_dict(torch.load(load_model_path))
        load_model.to(device)
    
    learning_rate = float(config.learning_rate)
    batch_size = config.batch_size
    patience = config.patience
    num_epochs = config.num_epochs
    model = LSTM_Transformer(pred_type='classify_deepset',
                            out_dim=n_classes,
                            n_offense=config.n_offense,
                            n_defense=config.n_defense,
                            hidden_dim=config.hidden_dim,
                            num_layers=config.num_layers,
                            nhead=config.nhead,
                            dropout=config.dropout).to(device) 
    
    if not load_date is None:
        params1 = load_model.named_parameters()
        params2 = model.named_parameters()
        
        dict_params2 = dict(params2)
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)    
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    
    best_loss = 9999
    best_epoch = 0
    # Train the model
    for epoch in range(num_epochs):
        permutation = torch.randperm(n_train)
        
        loss_avg = 0
        for i in range(0, n_train, batch_size):
            model.train()
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_x = Variable(torch.from_numpy(batch_x)).to(device)
            batch_y = Variable(torch.from_numpy(batch_y)).to(device)
            
            outs = model(batch_x.float())
            
            # obtain the loss function
            loss = criterion(outs, batch_y.long())
            
            loss.backward()
            optimizer.step()
            
            loss_avg += loss.item()*batch_x.shape[0]/n_train
        
        with torch.no_grad():
            idxs = torch.arange(n_val)
            correct = 0
            val_samples = 0
            val_loss_avg = 0
            for i in range(0, n_val, batch_size):
                model.eval()
                
                indices = idxs[i:i+batch_size]
                batch_x, batch_y = X_val[indices], y_val[indices]
                batch_x = Variable(torch.from_numpy(batch_x)).to(device)
                batch_y = Variable(torch.from_numpy(batch_y)).to(device)
                batch_y = batch_y.long().flatten()
                
                outs = model(batch_x.float())
                
                loss = criterion(outs, batch_y)
                val_loss_avg += loss.item()*batch_x.shape[0]/n_val
                
                arg_out = torch.argmax(outs, dim=-1)
                correct += (arg_out == batch_y).float().sum()
                val_samples += batch_y.shape[0]
            val_acc = correct / float(val_samples)
            if val_loss_avg < best_loss:
                print('best so far (saving):')
                best_loss = val_loss_avg
                best_epoch = epoch
                torch.save(model, save_folder+'/'+new_name+'.pth')
            if epoch - best_epoch > patience:
                break
        print(f'Epoch {epoch}: train_loss {loss_avg:0.4f} val_loss {val_loss_avg:0.4f} val_acc {100*val_acc:.2f}%')
    print('done. have a nice day!')

if __name__ == '__main__':
    finetune('./NFL_transformer_settings', '../data/middle.pkl', './saved_models', 'without_pretraining (36%)', 
              load_date=None, yaml_pretraining='NFL_trajectory_settings')
