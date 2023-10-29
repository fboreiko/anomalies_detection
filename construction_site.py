import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import pyro
import pyro.distributions as pyro_dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN, AutoRegressiveNN
from pyro.infer import SVI, Trace_ELBO

import tqdm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import optuna
import itertools
import os
import h5py


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = '9956'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class AutoregressiveSplineDeep(nn.Module):

    def __init__(self, hidden_dims, count_bins, bound, flow_length, learning_rate, input_dim=2, device='cpu'):
        super(AutoregressiveSplineDeep, self).__init__()
        self.base_dist = pyro_dist.Normal(torch.zeros(input_dim, device=device), torch.ones(input_dim, device=device))
        self.hypernet = AutoRegressiveNN(input_dim, 
                        hidden_dims, 
                        param_dims=[
                            count_bins,
                            count_bins,
                            (count_bins - 1),
                            count_bins,
                        ],
                    ).double()
        self.spline_transform = T.SplineAutoregressive(input_dim=input_dim, autoregressive_nn=self.hypernet, count_bins=count_bins, bound=bound)
        self.generative_flows = nn.ModuleList([self.spline_transform] * flow_length).to(device)
        self.flow_dist = pyro_dist.TransformedDistribution(self.base_dist, [self.spline_transform] * flow_length)
        self.optimizer = torch.optim.Adam(self.generative_flows.parameters(), lr=learning_rate)

    def model(self, X=None):
        N = len(X) if X is not None else None
        pyro.module("nf", self.generative_flows)
        with pyro.plate("data", N):
                obs = pyro.sample("obs", self.flow_dist, obs=X)

    def guide(self, X=None):
        pass

    def forward(self, z):
        zs = [z]
        for flow in ([self.spline_transform] * flow_length):
            z_i = flow(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def backward(self, x):
        zs = [x]
        for flow in [self.spline_transform] * flow_length:
            z_i = flow._inverse(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def sample(self, num_samples):
        z_0_samples = self.base_dist.sample([num_samples])
        zs, x = self.forward(z_0_samples)
        return x

    def log_prob(self, x):
        return self.flow_dist.log_prob(x)
    
    def save(self, destination_file=str, ntrial=int):
      
        torch.save(self.state_dict(), f'{destination_file}/trial{ntrial}-state_dict.pt')
        
        """
        torch.save(nn.ModuleList(modules), f'{destination_file}/trial{ntrial}-bygpu{gpu}-modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/trial{ntrial}-bygpu{gpu}-spline{i+1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/trial{ntrial}-bygpu{gpu}-modules_state_dict.pt')
        """
    def save_no_opt(self, destination_file=str):
        modules = self.generative_flows
        torch.save(nn.ModuleList(modules), f'{destination_file}/modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/spline{i+1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/modules_state_dict.pt')


class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, val_data, gpu_id: int, world_size: int, saving_file: str):
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = model.optimizer
        self.saving_file = saving_file
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_train_batch(self, inputs, training_loss):
        self.optimizer.zero_grad()
        loss = -self.model.module.flow_dist.log_prob(inputs).mean()
        training_loss += loss #loss.item()
        loss.backward()
        self.optimizer.step()
        #self.model.module.flow_dist.clear_cache()

        return training_loss

    def _run_valid_batch(self, inputs, valid_loss):
        loss = -self.model.module.flow_dist.log_prob(inputs).mean()
        valid_loss += loss #loss.item()
        #self.model.module.flow_dist.clear_cache()

        return valid_loss

    def _run_epoch(self, epoch):
        training_loss = 0
        for t, tinputs in enumerate(self.train_data):   
            tinputs = tinputs.to(self.gpu_id)
            training_loss = self._run_train_batch(tinputs, training_loss)
        #av_train_loss = running_loss/len(self.train_data)
        
        dist.barrier()
        torch.distributed.all_reduce(training_loss, op=dist.ReduceOp.AVG)
        
        with torch.no_grad():
            valid_loss = 0
            #for v, vinputs in enumerate(self.val_data):   
            val_data = self.val_data.to(self.gpu_id)
            valid_loss = self._run_valid_batch(val_data, valid_loss)
            #av_valid_loss = valid_loss/len(self.val_data)
        
        print(f'gpu {self.gpu_id} before: {valid_loss}')
        dist.barrier()
        torch.distributed.all_reduce(valid_loss, op=dist.ReduceOp.AVG)
        print(f'gpu {self.gpu_id} after: {valid_loss}')
                 
        return training_loss, valid_loss
    
    def _save_checkpoint(self, epoch, best_val_epoch, valid_loss, best_val_loss, ntrial):
        if valid_loss < best_val_loss:
            self.model.module.save(destination_file=self.saving_file, ntrial=ntrial)
            best_val_loss = valid_loss
            best_val_epoch = epoch
            return best_val_loss, best_val_epoch
        else: 
            return best_val_loss, best_val_epoch
    
    def train(self, num_epochs: int, ntrial: int):
        training_losses = []
        validation_losses = []
        best_val_loss = 100000 #torch.tensor([1000], dtype=torch.float64)
        best_val_epoch = 0
        if self.gpu_id == 0:
            epochs = tqdm.trange(num_epochs)
        else:
            epochs = range(num_epochs)
        for epoch in epochs:
            training_loss, valid_loss = self._run_epoch(epoch)
            training_losses.append(training_loss.item()*self.world_size)
            validation_losses.append(valid_loss.item()*self.world_size)
            best_val_loss, best_val_epoch = self._save_checkpoint(epoch, best_val_epoch, valid_loss.item()*self.world_size, best_val_loss, ntrial)
            if self.gpu_id == 0: 
                epochs.set_description("AV loss: {:.2f}, BV loss: {:f}, BV epoch: {:.0f}".format(training_losses[-1], best_val_loss, best_val_epoch))

        return training_losses, validation_losses, best_val_loss


def DataSplitOneD(data, val_ratio, test_ratio):
    val_size = int(np.size(data) * val_ratio)
    test_size = int(np.size(data) * test_ratio)
    train_set = data[:np.size(data)-val_size-test_size]
    val_set = data[np.size(data)-val_size-test_size:np.size(data)-test_size]
    test_set = data[np.size(data)-test_size:]
    return train_set, val_set, test_set


def GetMMstar(catalogues):
    # open the catalogue
    M_ent = []
    M_star = []
    for i in range(len(catalogues)):
        f = h5py.File(catalogues[i], 'r')

        M_ent = np.append(f['Subhalo/SubhaloMass'][:]*1e10, M_ent) #entire mass in Msun/h
        M_star = np.append(f['Subhalo/SubhaloMassType'][:,4]*1e10, M_star) #stellar masses in Msun/h
      
    M_ent = np.random.RandomState(seed=42).permutation(M_ent)
    M_star = np.random.RandomState(seed=42).permutation(M_star)
    
    # close file
    f.close()

    y_ent = np.log(1 + M_ent).astype(np.float64)
    y_star = np.log(1 + M_star).astype(np.float64)

    y_ent_norm = (y_ent - np.mean(y_ent))/ np.std(y_ent)
    y_star_norm = (y_star - np.mean(y_star))/ np.std(y_star)
    
    #optional
    #y_ent_norm = y_ent_norm/y_ent_norm.max()
    #y_star_norm = y_star_norm/y_star_norm.max()

    M_ent_trs, M_ent_vs, M_ent_ts = DataSplitOneD(y_ent_norm, val_ratio=0.08, test_ratio=0.08)
    M_star_trs, M_star_vs, M_star_ts = DataSplitOneD(y_star_norm, val_ratio=0.08, test_ratio=0.08)

    M_ent_trs, M_ent_vs, M_ent_ts = torch.from_numpy(M_ent_trs), torch.from_numpy(M_ent_vs), torch.from_numpy(M_ent_ts)
    M_star_trs, M_star_vs, M_star_ts = torch.from_numpy(M_star_trs), torch.from_numpy(M_star_vs), torch.from_numpy(M_star_ts)

    train_set, val_set, test_set = torch.stack((M_ent_trs, M_star_trs), dim=-1), torch.stack((M_ent_vs, M_star_vs), dim=-1), torch.stack((M_ent_ts, M_star_ts), dim=-1)

    return train_set, val_set, test_set


def load_train_objs(hidden_dims, count_bins, bound, flow_length, learning_rate, device):
    model = AutoregressiveSplineDeep(hidden_dims=hidden_dims, count_bins=count_bins, bound=bound, flow_length=flow_length, learning_rate=learning_rate, device=device)
    
    return model


def prepare_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=False, shuffle=False, sampler=DistributedSampler(dataset))


def worker(rank: int, world_size: int, catalogues: list, saving_destination: str, model_list: tuple, num_epochs: int, ntrial: int):
    
    ddp_setup(rank, world_size)
    hidden_dims1, hidden_dims2, count_bins, bound, flow_length, learning_rate = model_list
    train_set, val_set, test_set = GetMMstar(catalogues) 
    
    if rank == 0:
        print('train set size: ', train_set.size(0))
        print('val set size: ', val_set.size(0))
        print('test set size: ', test_set.size(0))
    
    train_data = prepare_dataloader(train_set, batch_size=512)
    val_data=val_set
    #val_data = prepare_dataloader(val_set, batch_size=512)
    
    if rank == 0:
        print('Number of validation batches: ', len(val_data))
    
    model = load_train_objs([hidden_dims1, hidden_dims2], count_bins, bound, flow_length, learning_rate, rank)
    trainer = Trainer(model, train_data, val_data, rank, world_size, saving_destination)
    training_losses, validation_losses, best_val_loss = trainer.train(num_epochs, ntrial)
    destroy_process_group()
    
    with open(f'{saving_destination}/trial{ntrial}-bestloss.txt', 'w') as f:
        f.write(str(best_val_loss))

    fig, ax = plt.subplots()
    #plt.plot(training_losses, 'b', label='training')
    plt.plot(validation_losses, 'r', label='validation')
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    #plt.yscale('log')
    plt.ylabel("Loss")
    plt.savefig(f'{saving_destination}/trial{ntrial}-loss.png')
    plt.close(fig)
    
    
    
    

