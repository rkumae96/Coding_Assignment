import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from your_model_file import GPT2, GPT2Config

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model, dataset, loss_fn, optimizer, batch_size, num_epochs):
    setup(rank, world_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(rank), targets.to(rank)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
    cleanup()

# Example usage in a script that can be run across multiple processes
