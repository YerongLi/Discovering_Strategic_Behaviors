import copy
import time
import pickle
import logging
import numpy as np
from scipy import stats, special
from collections import defaultdict
from sklearn.metrics import mean_squared_error

seed = 1
np.random.seed(seed)

strategy = 'cite'
start, end = 2000, 2018
fold = 0
Epochs = 20
Threshold = 5e-4

fpath = '/home/yuxinx2/DBLP'
logging.basicConfig(level=logging.INFO,filename=f'{fpath}_exp/TOT/run/{strategy}_{end}.log',filemode='a',format='%(asctime)s %(message)s',datefmt='%Y/%m/%d %I:%M:%S %p')
logging.info(f'Start Year {end} Fold {fold}')


def update(year):
    for each in curr_a_active:
        each_a_edgellh = curr_a_edgellh[curr_a_position[each]] 
        each_a_edgellh /= np.sum(each_a_edgellh,axis=1).reshape(-1,1)
        a_edgellhs[each].append(each_a_edgellh)
        a_timestamps[each].append([(year-start+1)/(end-start+2)]*len(each_a_edgellh))

a_edgellhs, a_timestamps = defaultdict(list), defaultdict(list)
for year in range(start, end):
    curr_a_active, curr_a_position, _, _, _, curr_a_edgellh = pickle.load(open(f'{fpath}/{strategy}_input/a_{strategy}_inputs_{year}', 'rb'))
    update(year)
curr_a_active, curr_a_position, _, _, _, curr_a_edgellh = pickle.load(open(f'{fpath}_exp/DDAN/{strategy}_input/a_{strategy}_inputs_{end}_{fold}', 'rb'))
update(end)

logging.info(f'Fold {fold}: Finish Loading')


T = 16 if strategy=='cite' else 8
S = 16 if strategy=='cite' else 8
alpha = 1/T
beta = 1/S
psi = np.ones((T,2))
beta_func = [special.beta(psi[t,0], psi[t,1]) for t in range(T)]

a_actives = sorted(a_timestamps.keys())
D = len(a_actives)

Y = [np.concatenate(a_timestamps[a],axis=0) for a in a_actives]
L = [np.concatenate(a_edgellhs[a],axis=0) for a in a_actives]

extra = 1
P = np.full((T,S), 1/(S+extra)) + np.identity(T)*(extra/(S+extra))
  
Z = [[] for _ in range(D)]
M, N = np.zeros((D,T)), np.zeros((T,S)) 
for node, edges in enumerate(Y):    
    topic_assign = np.random.randint(0,T,size=len(edges))
    topic, count = np.unique(topic_assign, return_counts=True)
    M[node][topic] = count    
    
    for edge in range(len(edges)):
        stra_assign = np.random.choice(S, p=P[topic_assign[edge]])
        Z[node].append([topic_assign[edge], stra_assign])
        N[topic_assign[edge], stra_assign] += 1
        
    Z[node] = np.array(Z[node])
        
logging.info(f'Fold {fold}: Finish Initialization')


prev_P = copy.deepcopy(P)
diff = mean_squared_error(P, np.zeros_like(P))

epoch = 0
start_time = time.time()
while abs(diff)>Threshold and epoch<Epochs:

    topic_Y = [[] for _ in range(T)]
    for node, edges in enumerate(Z):
        for edge, (old_topic, old_stra) in enumerate(edges):
        
            M[node][old_topic] -= 1
            N[old_topic][old_stra] -= 1
        
            topic_prob = np.log(np.matmul(P, L[node][edge].T))
            topic_prob += ((psi[:,0]-1)*np.log(1-Y[node][edge]) + (psi[:,1]-1)*np.log(Y[node][edge]) - np.log(beta_func))
            topic_prob += np.log(M[node]+alpha)
            topic_prob = np.exp(topic_prob + abs(max(topic_prob)))
            topic_prob /= np.sum(topic_prob)
            new_topic = np.random.choice(T, p=topic_prob)

            stra_prob = P[new_topic]*L[node][edge]
            stra_prob /= np.sum(stra_prob)
            new_stra = np.random.choice(S, p=stra_prob)

            Z[node][edge] = [new_topic, new_stra]
            M[node][new_topic] += 1
            N[new_topic][new_stra] += 1

            P[old_topic] = (N[old_topic]+beta) / (np.sum(N[old_topic])+beta*S)
            P[new_topic] = (N[new_topic]+beta) / (np.sum(N[new_topic])+beta*S)

        masks = [np.argwhere(Z[node][:,0]==i).flatten() for i in range(T)]
        for t in range(T):
            topic_Y[t].append(Y[node][masks[t]])

    topic_Y = [np.concatenate(each,axis=0) for each in topic_Y]
    stats_Y = np.array([[np.mean(curr_y), max(1e-2, np.var(curr_y))] for curr_y in topic_Y])
    psi_common = stats_Y[:,0]*(1-stats_Y[:,0])/stats_Y[:,1]-1
    psi = 1+np.vstack([stats_Y[:,0]*psi_common, (1-stats_Y[:,0])*psi_common]).T
    beta_func = [special.beta(psi[t,0], psi[t,1]) for t in range(T)]
    
    diff = mean_squared_error(prev_P, P)
    prev_P = copy.deepcopy(P)
    logging.info(f'Fold {fold}: Finish Epoch {epoch}, Diff {diff:.8f}')
    epoch += 1
    

results, targeting_year = {}, (end-start+1)/(end-start+2)
for node, edges in enumerate(Z):
    flag = False
    for edge, (topic, stra) in enumerate(edges):
        if Y[node][edge]==targeting_year:
            flag = True
            break
    if flag:
        topic_prob = np.log(M[node]+alpha)
        topic_prob += ((psi[:,0]-1)*np.log(1-targeting_year) + (psi[:,1]-1)*np.log(targeting_year) - np.log(beta_func))
        topic_prob = np.exp(topic_prob + abs(max(topic_prob)))
        topic_prob /= np.sum(topic_prob)
        results[node] = topic_prob
pickle.dump((results, P, Z, M, N, psi), open(f'{fpath}_exp/TOT/{strategy}_result/{strategy}_result_{end}_{fold}.pkl', 'wb'), -1)

logging.info(f'Finish Year {end} Fold {fold}: Time {time.time()-start_time}')
