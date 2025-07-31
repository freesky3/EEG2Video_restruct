'''use to process data and instruct dataloader
'''
import numpy as np
import torch

class myDataset(): 
    '''Generate dataset'''
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['features'].astype(np.float32), self.data[idx]['label'].astype(np.int64)-1




import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

def get_dataloader(data_path, batch_size, num_workers=0):
    '''Generate dataloader'''
    dataset = myDataset(data_path)
    trainset, validset = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        validset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, valid_loader
    