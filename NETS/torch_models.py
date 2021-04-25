from torch.autograd import Variable
import torch
import torch.nn as nn
import copy

device = 'cuda'

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        #h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc1(h_out[-1])
        out = out.relu()
        out = self.fc2(out)
        out = torch.sigmoid(out)
        
        return out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class my_TransformerEncoder(nn.Module):
    ''' shamelessly copy-pasted from nn.TransformerEncoder 
        adapted to get intermediate layer outputs 
    '''
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(my_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def get_n_transformer_layers(self):
        return self.num_layers

    def forward(self, src, layer_depth=-1, mask=None, src_key_padding_mask=None):
        output = src
        N_d = self.num_layers if layer_depth==-1 else layer_depth
        for i in range(N_d):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output
    
class LSTM_Transformer(nn.Module):
    def __init__(self, pred_type, out_dim, hidden_dim, 
                 n_offense, n_defense, num_layers=1, nhead=16, dropout=0.1):
        super(LSTM_Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.n_offense = n_offense
        self.n_defense = n_defense
        self.pred_type = pred_type
        self.out_dim = out_dim
        
        self.src_mask = None
        self.pos_encoder = nn.LSTM(input_size=2, hidden_size=hidden_dim-3,
                                   num_layers=1, batch_first=True)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = my_TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        
        if pred_type == 'trajectory':
            self.out_ball = nn.Linear(hidden_dim, out_dim)
            self.out_off = nn.Linear(hidden_dim, out_dim)
            self.out_def = nn.Linear(hidden_dim, out_dim)
        elif pred_type == 'classification':
            self.reduce_layer = nn.Linear(11*hidden_dim, hidden_dim)
            self.out_layer = nn.Linear(hidden_dim, out_dim)
        elif pred_type == 'classify_deepset':
            self.reduce_layer = nn.Linear(3*hidden_dim, hidden_dim)
            self.reduce_layerI = nn.Linear(hidden_dim, hidden_dim)
            self.reduce_layerII = nn.Linear(hidden_dim, hidden_dim)
            self.out_layer = nn.Linear(hidden_dim, out_dim)
        else:
            raise Exception(f'Type {pred_type} is not implemented!')
        
        #self.init_weights()
    def get_n_transformer_layers(self):
        return self.transformer_encoder.get_n_transformer_layers()
    
    def forward(self, x):
        (N, T, E) = x.shape
        output = self.get_embedding(x)
        
        if self.pred_type == 'classification':
            output = nn.ReLU() (self.reduce_layer( output ))
            output = self.out_layer( output )
            output = torch.sigmoid(output)
        elif self.pred_type == 'classify_deepset':
            output = nn.ReLU() (self.reduce_layer( output ))
            output = nn.ReLU() (self.reduce_layerI( output ))
            output = nn.ReLU() (self.reduce_layerII( output ))
            output = self.out_layer( output )
            if self.out_dim==1:
                output = torch.sigmoid(output)
            else:
                output = torch.softmax(output,dim=-1)
        elif self.pred_type == 'trajectory':
            out_b = self.out_ball( output[:, [0]] )
            out_o = self.out_off( output[:, 1:self.n_offense+1] )
            out_d = self.out_def( output[:, -self.n_defense:] )
            output = torch.cat( [out_b, out_o, out_d], dim=1 )
            
        return output

    def get_embedding(self, x, layer_depth=-1):
        # input shape (N, T, E)
        (N, T, E) = x.shape
        P = E//2
        x = x.reshape( (N, T, -1, 2) ) # shape (N, T, P, 2)
        x = x.permute(0, 2, 1, 3) # shape (N, P, T, 2)
        
        # run LSTM encoder
        x = x.reshape( (-1, T, 2) )
        x = self.pos_encoder(x)
        x = x[-1][0].reshape( (N, P, -1) )
        
        # add positional encoding
        ball = [1, 0, 0]
        off = [0, 1, 0]
        dev = [0, 0, 1]
        pos_e = torch.Tensor(ball + self.n_offense*off + self.n_defense*dev).to(x.get_device()).reshape( (-1, 3) )
        x = torch.cat( (x, pos_e.repeat(N, 1, 1)), dim=-1 )
        
        # input dimensions for transformer: (T, N, h_dim)
        x = x.permute(1, 0, 2)
        
        output = self.transformer_encoder(x, layer_depth=layer_depth) # output shape (N, P, h_dim)
        
        output = output.permute(1, 0, 2) # permute back
        if self.pred_type == 'classify_deepset':
            out_b = output[:, [0]]
            out_o = output[:, 1:self.n_offense+1].sum(1).unsqueeze(1)
            out_d = output[:, -self.n_defense:].sum(1).unsqueeze(1)
            out_combined = torch.cat( (out_b, out_o, out_d), 1 )
            
            output = out_combined.reshape( (N, -1) )
        elif self.pred_type == 'classification':
            output = output.reshape( (N, -1) )
        
        return output
