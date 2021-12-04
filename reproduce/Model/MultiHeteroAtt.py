import time
import pickle
import logging
import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Trainer import *


seed = 1
n_gpus = torch.cuda.device_count()
training_nlls = {'c_model':[], 'a_model':[]}


def train(trainer, rank, args):
    
    pass
    
def run(year, rank, world_size, args):
    
    if rank==0:
        logging.basicConfig(level=logging.INFO,filename=f'{args.path}/run/output_{args.strategy}.log',filemode='a',format='%(asctime)s %(message)s',datefmt='%Y/%m/%d %I:%M:%S %p')
        logging.info(f'Start Year {year} with {n_gpus} GPUs')
        
    trainer = Trainer(year, args.nstrategy, args.strategy, rank, world_size, "cuda:{}".format(rank), args.path)

    if rank==0: logging.info('Prepare Input')
    trainer.__prepare__(args.c_batchsize, args.a_batchsize)
    
    if rank==0: logging.info('Start Training')
    t = train(trainer, rank, args)
    if rank==0: logging.info('Finish Training, Time {}s'.format(t))
    
    # if rank==0: 
    #     logging.info('Save Results')
    #     save(trainer, year, args)
    #     logging.info('Finish Year {}'.format(year))
    #     logging.info('')
        

def init_process(rank, year, args):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    master_ip, master_port = '127.0.0.1', '12345'
    init_method = "tcp://{}:{}".format(master_ip, master_port)

    dist.init_process_group(backend='nccl', init_method=init_method, world_size=n_gpus, rank=rank)
    run(year, rank, n_gpus, args)

    
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--start_year', type=int, required=True)
    parser.add_argument('--end_year', type=int, required=True)
    parser.add_argument('--c_batchsize', type=int, required=True)
    parser.add_argument('--a_batchsize', type=int, required=True)
    parser.add_argument('--nstrategy', type=int, required=True)
    parser.add_argument('--strategy', type=str, required=True, choices=['cite','pub'])
    parser.add_argument('--threshold', default=2e-5, type=float)
    parser.add_argument('--max_nround', type=int, required=True)
    parser.add_argument('--min_nround', type=int, required=True)
    
    return parser.parse_args()
    

def main():
    
    args = parse_args()
    original_min_nround = args.min_nround
    for year in range(args.start_year, args.end_year):
        if year==2000: args.min_nround = int(original_min_nround/2)
        else: args.min_nround = original_min_nround           
        mp.spawn(init_process, args=(year,args), nprocs=n_gpus)
    
    
if __name__ == "__main__":
    main()    