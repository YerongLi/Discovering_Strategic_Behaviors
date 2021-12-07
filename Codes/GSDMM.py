import copy
import time
import pickle
import logging
import threading
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import os

seed = 1
np.random.seed(seed)

stra = 'cite'
year = 2010 #left: 2000, 2005, 2010, 2015, 2018
Epochs = 2
Threshold = 5e-4

folds = 5
thread_count = 5

fpath = '/home/yuxinx2/DBLP'
logging.basicConfig(level=logging.INFO,filename=f'./run/{stra}_{year}.log',filemode='a',format='%(asctime)s %(message)s',datefmt='%Y/%m/%d %I:%M:%S %p')
logging.info(f'Start Year {year}')

K = 16
L = 16
alpha = 1/K
beta = 1/L


def DMM(thread_id):
    
    logging.info(f'Start Fold {thread_id}')
    a_active, a_position, ac_adj, a_emb, da_emb, a_edgellh = pickle.load(open(f'{os.getenv("HOME")}/Discovering_Strategic_Behaviors/Dynamic_Dual_Attention_Networks/cite_input/a_cite_inputs_2018.pkl','rb'))
    
    # a_active, _, _, _, _, a_edgellh = pickle.load(open(f'{fpath}_exp/DDAN/{stra}_input/a_{stra}_inputs_{year}_{thread_id}', 'rb'))
    
    M = len(a_edgellh)
    N = np.sum([len(each) for each in a_edgellh])
    
    extra = 1
    Topics = np.full((K,L), 1/(L+extra)) + np.identity(K)*(extra/(L+extra))
    topic_norm = M-1 + alpha*K
    
    nodes_assign = np.random.randint(0,K,size=M)
    topic, count = np.unique(nodes_assign, return_counts=True)
    topics_assign_count = np.zeros(K, dtype=np.int32)
    topics_assign_count[topic] = count
    
    strategies_assign_count = np.zeros((K,L),dtype=np.int32)
    topic_stats = [stats.rv_discrete(values=(np.arange(L),Topics[i])) for i in range(K)]
    edges_assign = []
    for i in range(M):
        a_edgellh[i] /= np.sum(a_edgellh[i],axis=1).reshape(-1,1)
        edge_assign = topic_stats[nodes_assign[i]].rvs(size=len(a_edgellh[i]))
        strategy, count = np.unique(edge_assign, return_counts=True)
        strategies_assign_count[nodes_assign[i]][strategy] += count
        edges_assign.append(edge_assign)
    
    prev_Topics = copy.deepcopy(Topics)
    diff = mean_squared_error(Topics, np.zeros_like(Topics))
    
    epoch = 0
    start = time.time()
    while abs(diff)>Threshold and epoch<Epochs:
    
        for node in range(M):
        
            old_topic = nodes_assign[node]
            topics_assign_count[old_topic] -= 1
            old_strategy, old_count = np.unique(edges_assign[node], return_counts=True)
            strategies_assign_count[old_topic][old_strategy] -= old_count       
        
            topic_prob = np.sum(np.log(np.matmul(Topics, a_edgellh[node].T)),axis=1)        
            topic_prob = np.log((topics_assign_count+alpha)/topic_norm) + topic_prob
            topic_prob = np.exp(topic_prob + abs(max(topic_prob)))
            topic_prob = topic_prob / np.sum(topic_prob)
            new_topic = np.random.choice(K, p=topic_prob)
            topics_assign_count[new_topic] += 1
            nodes_assign[node] = new_topic
        
            strategy_prob = Topics[new_topic]*a_edgellh[node]
            strategy_prob = strategy_prob / np.sum(strategy_prob,axis=1).reshape(-1,1)    
            new_strategies = np.array([np.random.choice(L, p=strategy_prob[i]) for i in range(len(a_edgellh[node]))])
            edges_assign[node] = new_strategies
            new_strategy, new_count = np.unique(new_strategies, return_counts=True)
            strategies_assign_count[new_topic][new_strategy] += new_count      
        
            Topics[new_topic] = (strategies_assign_count[new_topic]+beta) / (np.sum(strategies_assign_count[new_topic])+beta*L)
            Topics[old_topic] = (strategies_assign_count[old_topic]+beta) / (np.sum(strategies_assign_count[old_topic])+beta*L)
        
        diff = mean_squared_error(prev_Topics, Topics)
        prev_Topics = copy.deepcopy(Topics)
        logging.info(f'Fold {thread_id}, Epoch {epoch}, Diff {diff:.8f}')
        epoch += 1
        
    results = {}
    for i, a in enumerate(a_active):
        results[a] = Topics[nodes_assign[i]]
    pickle.dump((results, Topics, nodes_assign, topics_assign_count, edges_assign, strategies_assign_count), open(f'./{stra}_result/{stra}_result_{year}_{thread_id}.pkl', 'wb'), -1)
    
    logging.info(f"Finish Fold {thread_id}, Time {time.time()-start}")
    
    
if __name__ == "__main__":
    threads = []
    for i in range(thread_count):
        threads.append(threading.Thread(target=DMM, args=(i,)))
 
    for i in range(thread_count):
        threads[i].start()
  
    for i in range(thread_count):
        threads[i].join()
        
    logging.info(f'Finish Year {year}')