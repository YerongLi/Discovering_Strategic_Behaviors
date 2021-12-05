import pickle
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from utils import *
import tqdm

class LogisticRegression(nn.Module):
    
    def __init__(self, dimension, device):
        super(LogisticRegression, self).__init__()
        
        self.W = nn.Parameter(torch.zeros(size=(dimension, 1))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.BCELoss = nn.BCELoss()
        
    def forward(self, X):        
        
        return torch.sigmoid(torch.matmul(X, F.softmax(self.W, dim=0)))
    
    def loss(self, X, y):
        
        output = self.forward(X)
        return self.BCELoss(output.squeeze(), y.squeeze())
    
    def softmax(self):
        
        return F.softmax(self.W, dim=0).squeeze().detach().cpu().numpy()


def LR(stra):
    
    for fold in range(Folds):
        # exp_input = pickle.load(open(f'/home/yuxinx2/DBLP_exp/LR/{stra}/{year}_{stra}_input_{fold}.pkl','rb'))

        a_active, a_position, ac_adj, a_emb, da_emb, a_edgellh = pickle.load(open(f'{os.getenv("HOME")}/yerong/Discovering_Strategic_Behaviors/Dynamic_Dual_Attention_Networks/cite_input/a_cite_inputs_2018.pkl','rb'))
        exp_input = zip(a_active, a_edgellh)      
        results = {}
        # for author, llhs in exp_input.items():
        for author, llhs in exp_input:
                
            X = torch.from_numpy(llhs/np.sum(llhs,axis=1).reshape(-1,1)).float().to('cpu')
            y = torch.from_numpy(np.ones(len(llhs))).float().to('cpu')
                
            dim = 16 if stra=='cite' else 8
            model = LogisticRegression(dim, 'cpu')
            optimizer = optim.Adam(model.parameters(), lr=0.5, weight_decay=5e-4)
                
            prev_loss = np.inf
            for i in tqdm.tqdm(range(10)):
                model.train()
                curr_loss = model.loss(X, y)
                curr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if curr_loss >= prev_loss: break
                else: prev_loss = curr_loss
                        
            results[author] = model.softmax()
                
        pickle.dump(results, open(f'./yerong.pkl','wb'), -1)

        
year = 2015
Folds = 5
Strategies = ['cite']
# Strategies = ['cite', 'pub']
            
if __name__ == "__main__":
    threads = []
    for stra in Strategies:
        threads.append(threading.Thread(target=LR, args=(stra,)))
 
    for thread in threads:
        thread.start()
  
    for thread in threads:
        thread.join()