# Defining Pyro model architectures and their corresponding loading functions

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import DenseNN, AutoRegressiveNN

import numpy as np
import itertools


class SimpleSpline(nn.Module):
    # Gaussian normal base distribution, Batch Normalization and flow_length number of Spline layers
    def __init__(self, input_dim=2, count_bins=int, bound=int, flow_length=int, device='cpu'):
        super(SimpleSpline, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim, device=device), torch.ones(input_dim, device=device))
        self.bn_transform = T.BatchNorm(input_dim).to(device)
        self.spline_transforms = [T.Spline(input_dim, count_bins, bound).to(device) for _ in range(flow_length)]
        self.generative_flows = [self.bn_transform] + self.spline_transforms
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.generative_flows)

    def model(self, X=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.spline_transforms))
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

    def train(self, input, optimizer):
        optimizer.zero_grad()
        loss = -self.flow_dist.log_prob(input).mean()
        loss.backward()
        optimizer.step()
        self.flow_dist.clear_cache()
        return loss

    def validate(self, input):
        val_loss = -self.flow_dist.log_prob(input).mean()
        self.flow_dist.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/trial{ntrial}-modules.pt')
        torch.save(self.bn_transform, f'{destination_file}/trial{ntrial}-batchnorm.pt')
        for i in range(len(nn.ModuleList(self.generative_flows)) - 1):
            torch.save(self.spline_transforms[i], f'{destination_file}/trial{ntrial}-spline{i + 1}.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(),
                   f'{destination_file}/trial{ntrial}-modules_state_dict.pt')


def loadSimpleSpline(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    generative_flows = []
    generative_flows.append(torch.load(f'{source_file}/trial{ntrial}-batchnorm.pt', map_location=device))
    for i in range(len(modules) - 1):
        generative_flows.append(torch.load(f'{source_file}/trial{ntrial}-spline{i + 1}.pt', map_location=device))
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, generative_flows, state_dict


class ConditionalSpline(nn.Module):
    # Gaussian normal base distribution, Conditional Spline layer
    def __init__(self, hidden_dims=tuple, count_bins=int, bound=int, device='cpu'):
        super(ConditionalSpline, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device))
        self.x1_transform = T.Spline(1, count_bins=count_bins, bound=bound).to(device)
        self.dist_x1 = dist.TransformedDistribution(self.base_dist, [self.x1_transform])
        self.x2_transform = T.conditional_spline(1, context_dim=1, hidden_dims=hidden_dims, count_bins=count_bins,
                                                 bound=bound).to(device)
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(self.base_dist, [self.x2_transform])
        self.generative_flows = [self.x1_transform, self.x2_transform]

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

    def train(self, x1, x2, optimizer):
        optimizer.zero_grad()
        ln_p_x1 = self.dist_x1.log_prob(x1)
        ln_p_x2_given_x1 = self.dist_x2_given_x1.condition(x1.detach()).log_prob(x2.detach())
        loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        # loss = -ln_p_x2_given_x1.mean()
        loss.backward()
        optimizer.step()
        self.dist_x1.clear_cache()
        self.dist_x2_given_x1.clear_cache()
        return loss

    def validate(self, x1, x2):
        ln_p_x1 = self.dist_x1.log_prob(x1)
        ln_p_x2_given_x1 = self.dist_x2_given_x1.condition(x1.detach()).log_prob(x2.detach())
        val_loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        self.dist_x1.clear_cache()
        self.dist_x2_given_x1.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/trial{ntrial}-modules.pt')
        torch.save(self.x1_transform, f'{destination_file}/trial{ntrial}-x1_transform.pt')
        torch.save(self.x2_transform, f'{destination_file}/trial{ntrial}-x2_transform.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(),
                   f'{destination_file}/trial{ntrial}-modules_state_dict.pt')


def loadConditionalSpline(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    x1_transform = torch.load(f'{source_file}/trial{ntrial}-x1_transform.pt', map_location=device)
    x2_transform = torch.load(f'{source_file}/trial{ntrial}-x2_transform.pt', map_location=device)
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, x1_transform, x2_transform, state_dict


class ConditionalSplineBatchNorm(nn.Module):  # useful for self-defined toy dataset

    def __init__(self, count_bins=int, bound=int, device='cpu'):
        super(ConditionalSplineBatchNorm, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device))
        self.batchnorm = T.BatchNorm(1).to(device)
        self.x1_transform = T.Spline(1, count_bins=count_bins, bound=bound).to(device)
        self.dist_x1 = dist.TransformedDistribution(self.base_dist, [self.batchnorm, self.x1_transform])
        self.x2_transform = T.conditional_spline(1, context_dim=1, count_bins=count_bins, bound=bound).to(device)
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(self.base_dist, [self.x2_transform])
        self.generative_flows = [self.batchnorm, self.x1_transform, self.x2_transform]

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

    def train(self, x1, x2, optimizer):
        optimizer.zero_grad()
        ln_p_x1 = self.dist_x1.log_prob(x1)
        ln_p_x2_given_x1 = self.dist_x2_given_x1.condition(x1.detach()).log_prob(x2.detach())
        loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        loss.backward()
        optimizer.step()
        self.dist_x1.clear_cache()
        self.dist_x2_given_x1.clear_cache()
        return loss

    def validate(self, x1, x2):
        ln_p_x1 = self.dist_x1.log_prob(x1)
        ln_p_x2_given_x1 = self.dist_x2_given_x1.condition(x1.detach()).log_prob(x2.detach())
        val_loss = -(ln_p_x1 + ln_p_x2_given_x1).mean()
        self.dist_x1.clear_cache()
        self.dist_x2_given_x1.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/trial{ntrial}-modules.pt')
        torch.save(self.batchnorm, f'{destination_file}/trial{ntrial}-batchnorm.pt')
        torch.save(self.x1_transform, f'{destination_file}/trial{ntrial}-x1_transform.pt')
        torch.save(self.x2_transform, f'{destination_file}/trial{ntrial}-x2_transform.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(),
                   f'{destination_file}/trial{ntrial}-modules_state_dict.pt')


def loadConditionalSplineBatchNorm(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    batchnorm = torch.load(f'{source_file}/trial{ntrial}-batchnorm.pt', map_location=device)
    x1_transform = torch.load(f'{source_file}/trial{ntrial}-x1_transform.pt', map_location=device)
    x2_transform = torch.load(f'{source_file}/trial{ntrial}-x2_transform.pt', map_location=device)
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, batchnorm, x1_transform, x2_transform, state_dict


class CouplingSpline(nn.Module):  # converges - don't know how to solve this

    def __init__(self, input_dim=2, split_dim=1, hidden_dims=tuple, count_bins=int, bound=int, flow_length=int,
                 device='cpu'):
        super(CouplingSpline, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim, dtype=torch.float64, device=device),
                                     torch.ones(input_dim, dtype=torch.float64, device=device))
        self.nn = DenseNN(split_dim,
                          hidden_dims,
                          param_dims=[
                              (input_dim - split_dim) * count_bins,
                              (input_dim - split_dim) * count_bins,
                              (input_dim - split_dim) * (count_bins - 1),
                              (input_dim - split_dim) * count_bins,
                          ],
                          ).double()
        self.spline_transforms = [
            T.SplineCoupling(input_dim=input_dim, split_dim=split_dim, hypernet=self.nn, count_bins=count_bins,
                             bound=bound).to(device) for _ in range(flow_length)]
        # self.bns = [T.BatchNorm(input_dim=2) for _ in range(flow_length)]
        self.perms = [T.permute(input_dim=input_dim) for _ in range(flow_length)]
        self.generative_flows = self.spline_transforms
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.generative_flows)

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

    def train(self, input, optimizer):
        optimizer.zero_grad()
        loss = -self.flow_dist.log_prob(input).mean()
        loss.backward()
        optimizer.step()
        self.flow_dist.clear_cache()
        return loss

    def validate(self, input):
        val_loss = -self.flow_dist.log_prob(input).mean()
        self.flow_dist.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        # modules = list(itertools.chain(*zip(self.spline_transforms, self.bns)))[:-1]
        modules = self.spline_transforms
        torch.save(nn.ModuleList(modules), f'{destination_file}/trial{ntrial}-modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/trial{ntrial}-spline{i + 1}.pt')
            # else:
            # torch.save(modules[i], f'{destination_file}/trial{ntrial}-batchnorm{i // 2 + 1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/trial{ntrial}-modules_state_dict.pt')


def loadCouplingSpline(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    spline_transforms = []
    # bns = []
    for i in range(len(modules)):
        spline_transforms.append(torch.load(f'{source_file}/trial{ntrial}-spline{i + 1}.pt', map_location=device))
        # else:
        # bns.append(torch.load(f'{source_file}/trial{ntrial}-batchnorm{i // 2 + 1}.pt', map_location=device))
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, spline_transforms, state_dict


class AutoregressiveSpline(nn.Module):

    def __init__(self, input_dim=2, hidden_dims=tuple, count_bins=int, bound=int, device='cpu'):
        super(AutoregressiveSpline, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim, device=device), torch.ones(input_dim, device=device))
        self.hypernet = AutoRegressiveNN(input_dim,
                                         hidden_dims,
                                         param_dims=[
                                             count_bins,
                                             count_bins,
                                             (count_bins - 1),
                                             count_bins,
                                         ],
                                         ).double().to(device)
        self.spline_transforms = T.SplineAutoregressive(input_dim=input_dim, autoregressive_nn=self.hypernet,
                                                        count_bins=count_bins, bound=bound).to(device)
        self.flow_dist = dist.TransformedDistribution(self.base_dist, [self.spline_transforms])
        self.generative_flows = [self.spline_transforms]

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

    def train(self, input, optimizer):
        optimizer.zero_grad()
        loss = -self.flow_dist.log_prob(input).mean()
        # loss = -ln_p_x2_given_x1.mean()
        loss.backward()
        optimizer.step()
        self.flow_dist.clear_cache()
        return loss

    def validate(self, input):
        val_loss = -self.flow_dist.log_prob(input).mean()
        self.flow_dist.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/trial{ntrial}-modules.pt')
        torch.save(self.spline_transforms, f'{destination_file}/trial{ntrial}-spline.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(),
                   f'{destination_file}/trial{ntrial}-modules_state_dict.pt')

    def save_no_opt(self, destination_file=str):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/modules.pt')
        torch.save(self.spline_transforms, f'{destination_file}/spline.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(), f'{destination_file}/modules_state_dict.pt')


def loadAutoregressiveSpline(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    spline_transforms = torch.load(f'{source_file}/trial{ntrial}-spline.pt', map_location=device)
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, spline_transforms, state_dict


class AutoregressiveSplineDeep(nn.Module):

    def __init__(self, input_dim=2, hidden_dims=tuple, count_bins=int, bound=int, flow_length=2, device='cpu'):
        super(AutoregressiveSplineDeep, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim, device=device), torch.ones(input_dim, device=device))
        self.hypernet = AutoRegressiveNN(input_dim,
                                         hidden_dims,
                                         param_dims=[
                                             count_bins,
                                             count_bins,
                                             (count_bins - 1),
                                             count_bins,
                                         ],
                                         ).double()
        self.spline_transforms = [
            T.SplineAutoregressive(input_dim=input_dim, autoregressive_nn=self.hypernet, count_bins=count_bins,
                                   bound=bound).to(device) for _ in range(flow_length)]
        self.generative_flows = self.spline_transforms
        self.flow_dist = dist.TransformedDistribution(self.base_dist, self.generative_flows)

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

    def train(self, input, optimizer):
        optimizer.zero_grad()
        loss = -self.flow_dist.log_prob(input).mean()
        # loss = -ln_p_x2_given_x1.mean()
        loss.backward()
        optimizer.step()
        self.flow_dist.clear_cache()
        return loss

    def validate(self, input):
        val_loss = -self.flow_dist.log_prob(input).mean()
        self.flow_dist.clear_cache()
        return val_loss

    def save(self, destination_file=str, ntrial=int):
        modules = self.generative_flows
        torch.save(nn.ModuleList(modules), f'{destination_file}/trial{ntrial}-modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/trial{ntrial}-spline{i + 1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/trial{ntrial}-modules_state_dict.pt')

    def save_no_opt(self, destination_file=str):
        modules = self.generative_flows
        torch.save(nn.ModuleList(modules), f'{destination_file}/modules.pt')
        for i in range(len(modules)):
            torch.save(modules[i], f'{destination_file}/spline{i + 1}.pt')
        torch.save(nn.ModuleList(modules).state_dict(), f'{destination_file}/modules_state_dict.pt')


def loadAutoregressiveSplineDeep(source_file=str, ntrial=int, gpu=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-bygpu{gpu}-modules.pt', map_location=device)
    spline_transforms = []
    # bns = []
    for i in range(len(modules)):
        spline_transforms.append(
            torch.load(f'{source_file}/trial{ntrial}-bygpu{gpu}-spline{i + 1}.pt', map_location=device))
        # else:
        # bns.append(torch.load(f'{source_file}/trial{ntrial}-batchnorm{i // 2 + 1}.pt', map_location=device))
    state_dict = torch.load(f'{source_file}/trial{ntrial}-bygpu{gpu}-modules_state_dict.pt', map_location=device)

    return modules, spline_transforms, state_dict


# Following is useful for Spline-BatchNorm
'''def save(self, destination_file=str, ntrial=int):
        torch.save(nn.ModuleList(self.generative_flows), f'{destination_file}/trial{ntrial}-modules.pt')
        for i in range(len(nn.ModuleList(self.generative_flows))):
            if i % 2 == 0:
                torch.save(self.generative_flows[i], f'{destination_file}/trial{ntrial}-spline{i // 2 + 1}.pt')
            else:
                torch.save(self.generative_flows[i], f'{destination_file}/trial{ntrial}-permutation{i // 2 + 1}.pt')
        torch.save(nn.ModuleList(self.generative_flows).state_dict(), f'{destination_file}/trial{ntrial}-modules_state_dict.pt')'''
'''def loadCouplingSpline(source_file=str, ntrial=int, device='cpu'):
    modules = torch.load(f'{source_file}/trial{ntrial}-modules.pt', map_location=device)
    generative_flows = []
    for i in range(len(modules)):
        if i % 2 == 0:
            generative_flows.append(torch.load(f'{source_file}/trial{ntrial}-spline{i // 2 + 1}.pt', map_location=device))
        else:
            generative_flows.append(torch.load(f'{source_file}/trial{ntrial}-permutation{i // 2 + 1}.pt', map_location=device))
    state_dict = torch.load(f'{source_file}/trial{ntrial}-modules_state_dict.pt', map_location=device)

    return modules, generative_flows, state_dict'''
