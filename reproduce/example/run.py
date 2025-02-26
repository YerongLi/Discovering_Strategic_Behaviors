"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""Blocking point-to-point communication."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         print('Rank 0 started sending')
#         dist.send(tensor=tensor, dst=1)
#         tensor -=2
#         dist.send(tensor=tensor, dst=2)
#     elif rank == 1:
#         # Receive tensor from process 0
#         print('Rank 1 started receiving')
#         dist.recv(tensor=tensor, src=0)
#         tensor += 1
#     else:
#         print('Rank 2 started receiving')
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()