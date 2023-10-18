import os
import hostlist
import time
import random
import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data import create_dataset, custom_collate_fn


class Engine(object):
    def __init__(self, opt):
        self.opt = opt

        # multi node mode on slurm cluster
        if 'SLURM_JOB_NUM_NODES' in os.environ:
	    node_id = int(os.environ['SLURM_NODEID'])
	    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
            self.global_rank = int(os.environ['RANK'])
        else:
            node_id = 0
            n_nodes = 1
            self.global_rank = int(os.environ['LOCAL_RANK'])

        self.distributed = True

        if self.distributed:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(0, 300))
            self.is_main = node_id == 0 and self.local_rank == 0
        else:
            self.local_rank = 0
            self.world_size = 1
            self.is_main = True

        # wait a little while so that all processes do not print at once
        time.sleep(self.global_rank * 0.1)
        print(f"Initializing node {node_id + 1} / {n_nodes}, rank {self.local_rank + 1} (local) {self.global_rank + 1} (global) / {self.world_size}")

    def data_parallel(self, model, broadcast_buffers=True):
        if self.distributed and model is not None:
            model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=broadcast_buffers)
        return model

    def create_dataset(self, opt, phase="train", fold=None, load_vid=False, load_rgb=True):
        opt = opt["base"] if type(opt) is dict else opt
        verbose = self.is_main
        return create_dataset(opt, phase=phase, fold=fold, load_vid=load_vid, load_rgb=load_rgb, verbose=verbose)

    def create_dataloader(self, dataset, batch_size=None, num_workers=1, is_train=True, persist=False):
        datasampler = None
        is_shuffle = is_train or self.opt.shuffle_valid
        drop_last = True
        pin_memory = not persist
        if self.distributed:
            # is_shuffle = False
            datasampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_shuffle)
            batch_size = batch_size // self.world_size
            is_shuffle = False

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 drop_last=drop_last,
                                                 shuffle=is_shuffle,
                                                 pin_memory=pin_memory,
                                                 sampler=datasampler,
                                                 worker_init_fn=lambda _: np.random.seed(),
                                                 collate_fn=custom_collate_fn,
                                                 persistent_workers=persist)

        return dataloader, datasampler, batch_size

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return tensor

    def all_gather_tensor(self, tensor):
        if self.distributed:
            tensor_list = [torch.ones_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, tensor)
            return tensor_list
        else:
            return [tensor]

    def barrier(self):
        dist.barrier()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            print("An exception occurred during Engine initialization, give up running process")
            return False

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)
    return tensor

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
