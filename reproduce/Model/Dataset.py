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
# Get the number of GPUs
n_gpus = torch.cuda.device_count()
training_nlls = {'c_model':[], 'a_model':[]}


def train(trainer, rank, args):
    pass


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

def run(year, rank, world_size, args):
    pass

def init_process(rank, year, args):
    """ Initialize the distributed environment. """
    
    
    master_ip, master_port = '127.0.0.1', '12345'
    init_method = "tcp://{}:{}".format(master_ip, master_port)

    dist.init_process_group(backend='', rank=rank, world_size=size)
    
    run(year, rank, n_gpus, args)

def main():
    
    args = parse_args()
    original_min_nround = args.min_nround
    for year in range(args.start_year, args.end_year):
        if year==2000: args.min_nround = int(original_min_nround/2)
        else: args.min_nround = original_min_nround           
        mp.spawn(init_process, args=(year,args), nprocs=n_gpus)
    
    