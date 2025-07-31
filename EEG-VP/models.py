'''
define the models here

models:
    mlpnet: basic net
    glfnet: used to get the global and local features of EEG signals
'''
import torch.nn as nn
import torch

class mlpnet(nn.Module):
    '''used to define basic net
    todo: specify the usage, what is the meaning of input_dim and out_dim
    args: 
        out_dim: todo
        input_dim: todo
    '''
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):
        # x, which size is [batch_size, C, 5]
        out = self.net(x)
        return out




class glfnet(nn.Module):
    '''glfnet used to get the global and local features of EEG signals
    todo: specify the usage, why we bother to use two nets to get global and local features
    args: 
        out_dim and input_dim for mlpnet
        emb_dim: todo, what's the meaning of this parameter?
    '''
    def __init__(self, out_dim, emb_dim, input_dim):
        super(glfnet, self).__init__()
        self.globalnet = mlpnet(emb_dim, input_dim)
        self.occipital_index = list(range(50*5, 62*5))
        self.occipital_localnet = mlpnet(emb_dim, 12*5)
        self.linearnet = nn.Linear(emb_dim*2, out_dim)

    def forward(self, x):
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.linearnet(torch.cat((global_feature, occipital_feature), 1))
        return out