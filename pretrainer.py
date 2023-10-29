#!/mnt/home/fboreiko/home/fboreiko/bin/python
#SBATCH -p gpu 
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH -t 24:00:00

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import DenseNN, AutoRegressiveNN
import csv
import matplotlib.pyplot as plt
import itertools
import h5py
from models import loadConditionalSpline, loadConditionalSplineBatchNorm, loadSimpleSpline, loadCouplingSpline, loadAutoregressiveSpline, loadAutoregressiveSplineDeep
from datasampler import DistOneD, DistTwoD, DatasetMoons, TwoCircles
from construction_site import GetMMstar
import pandas as pd
import seaborn as sns
import tqdm
from scipy import stats
import os

class AutoregressiveSplineDeep(nn.Module):

    def __init__(self, hidden_dims, count_bins, bound, flow_length, learning_rate, input_dim=2, device='cpu'):
        super(AutoregressiveSplineDeep, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim, device=device, dtype=torch.float64), torch.ones(input_dim, device=device, dtype=torch.float64))
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
        self.flow_dist = dist.TransformedDistribution(self.base_dist, [self.spline_transform] * flow_length)
        self.optimizer = torch.optim.Adam(self.generative_flows.parameters(), lr=learning_rate)

    def model(self, X=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.generative_flows))
        with pyro.plate("data", N):
                obs = pyro.sample("obs", self.flow_dist, obs=X)

    def guide(self, X=None):
        pass

    def forward(self, z):
        zs = [z]
        for flow in self.generative_flows:
            z_i = flow(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def backward(self, x):
        zs = [x]
        for flow in self.normalizing_flows:
            z_i = flow._inverse(zs[-1])
            zs.append(z_i)
        return zs, z_i

    def sample(self, num_samples):
        z_0_samples = self.base_dist.sample([num_samples])
        zs, x = self.forward(z_0_samples)
        return x

    def log_prob(self, x):
        return self.flow_dist.log_prob(x)

    def train(self, input):
        self.optimizer.zero_grad()
        loss = -self.flow_dist.log_prob(input).mean()
        #loss = -ln_p_x2_given_x1.mean()
        loss.backward()
        self.optimizer.step()
        self.flow_dist.clear_cache()
        return loss

    def validate(self, input):
        val_loss = -self.flow_dist.log_prob(input).mean()
        self.flow_dist.clear_cache()
        return val_loss
    
    def save(self, destination_file=str, ntrial=int):
        torch.save(self.state_dict(), f'{destination_file}/trial{ntrial}-state_dict.pt')
        
    def save_no_opt(self, destination_file=str):
        modules = self.generative_flows
        torch.save(nn.ModuleList(modules), f'{destination_file}/modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/spline{i+1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/modules_state_dict.pt')
        
        
model_name = 'AS(CustomNN)_singleGPU'
d_name = 'MulMMstar'
catalogue1 = 'catalogues/fof_subhalo_tab_030.hdf5'
catalogue2 = 'catalogues/fof_subhalo_tab_031.hdf5'
catalogue3 = 'catalogues/fof_subhalo_tab_032.hdf5'
catalogue4 = 'catalogues/fof_subhalo_tab_033.hdf5'
catalogues = [catalogue1, catalogue2, catalogue3, catalogue4]
train_set, val_set, test_set = GetMMstar(catalogues)

print('train set size: ', train_set.size(0))
print('val set size: ', val_set.size(0))
print('test set size: ', test_set.size(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

ntrial = 55   #change made
raw = 47
destination_file = 'results'

params = pd.read_csv(f'{destination_file}/trials-info.csv')
hidden_dims1 = params['hidden_dims1'][raw]
hidden_dims2 = params['hidden_dims2'][raw]
count_bins = params['count_bins'][raw]
bound = params['bound'][raw]
flow_length = params['flow_length'][raw]
learning_rate = params['learning_rate'][raw]
best_val_loss = params['best_val_loss'][raw]

model = AutoregressiveSplineDeep(hidden_dims=[hidden_dims1, hidden_dims2], count_bins=count_bins, bound=bound, flow_length=flow_length, learning_rate=learning_rate, device=device)
model.load_state_dict(torch.load(f'{destination_file}/trial{ntrial}-state_dict.pt',  map_location=device))

num_epochs = 1000
batch_size = 128

trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
#validloader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

training_losses = []
validation_losses = []
best_val_loss = 1e7
best_val_epoch = 0

epochs = tqdm.trange(num_epochs)
for epoch in epochs:

    #Train
    model.generative_flows.train()

    running_loss = 0
    for _, train_batch in enumerate(trainloader):
        #x1 = train_batch[:,0][:,None].to(device)
        #x2 = train_batch[:,1][:,None].to(device)
        train_batch = train_batch.to(device)
        loss = model.train(train_batch)
        running_loss += loss
    training_losses.append(running_loss.item()) #change made

    #Val
    with torch.no_grad():
        model.generative_flows.eval()
        #val_loss_cumul = 0
        #for _, val_batch in enumerate(validloader):
        #x1 = val_batch[:,0][:,None].to(device)
        #x2 = val_batch[:,1][:,None].to(device)
        val_set = val_set.to(device)
        val_loss = model.validate(val_set)
        #val_loss_cumul += val_loss
        validation_losses.append(val_loss.item())
        if val_loss.item() < best_val_loss:  #change made
            best_val_loss = val_loss.item()  #change made
            best_val_epoch = epoch

            #Saving the model. /content/results
            model.save(destination_file='pretrainer_results', ntrial=ntrial)
            
    epochs.set_description("AV loss: {:.2f}, BV loss: {:f}, BV epoch: {:.0f}".format(training_losses[-1], best_val_loss, best_val_epoch))

fig, ax = plt.subplots()
#plt.plot(training_losses, 'b', label='training')
plt.plot(validation_losses, 'r', label='validation')
plt.legend(loc="upper right")
plt.xlabel("Epochs")
#plt.yscale('log')
plt.ylabel("Loss")
plt.savefig(f'pretrainer_results/trial{ntrial}-loss.png')
plt.close(fig)

print('Finished!')