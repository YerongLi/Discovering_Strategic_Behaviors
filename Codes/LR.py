import pickle
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
        return self.BCELoss(output.squeeze(), y)
    
    def softmax(self):
        
        return F.softmax(self.W, dim=0).squeeze().detach().cpu().numpy()


def LR(stra):
    
    for fold in range(Folds):
        exp_input = pickle.load(open(f'/home/yuxinx2/DBLP_exp/LR/{stra}/{year}_{stra}_input_{fold}.pkl','rb'))
            
        results = {}
        for author, llhs in exp_input.items():
                
            X = torch.from_numpy(llhs/np.sum(llhs,axis=1).reshape(-1,1)).float().to('cpu')
            y = torch.from_numpy(np.ones(len(llhs))).float().to('cpu')
                
            dim = 16 if stra=='cite' else 8
            model = LogisticRegression(dim, 'cpu')
            optimizer = optim.Adam(model.parameters(), lr=0.5, weight_decay=5e-4)
                
            prev_loss = np.inf
            while True:
                model.train()
                curr_loss = model.loss(X, y)
                curr_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if curr_loss >= prev_loss: break
                else: prev_loss = curr_loss
                        
            results[author] = model.softmax()
                
        pickle.dump(results, open(f'/home/yuxinx2/DBLP_exp/LR/{stra}_result/{stra}_result_{year}_{fold}.pkl','wb'), -1)

        
year = 2015
Folds = 5
Strategies = ['cite', 'pub']
            
if __name__ == "__main__":
    threads = []
    for stra in Strategies:
        threads.append(threading.Thread(target=LR, args=(stra,)))
 
    for thread in threads:
        thread.start()
  
    for thread in threads:
        thread.join()