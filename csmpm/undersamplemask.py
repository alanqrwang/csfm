import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt

from . import layers, utils

class NormalMask(nn.Module):
    def __init__(self, mask_type, image_dims, device):
        super(NormalMask, self).__init__()
        
        self.image_dims = image_dims
        self.type = mask_type
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.pmask_slope = 0.5

        if self.type == 'learned':
            # Mean mask
            self.mmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            self.mmask.requires_grad = True
            self.mmask.data.uniform_(0, 1)
            self.mmask.register_hook(lambda grad: print('1', grad))

            # Variance mask
            # self.vmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            # self.vmask.requires_grad = True
            # self.vmask.data.uniform_(eps, 1-eps)
            # self.vmask.data = -torch.log(1. / self.vmask.data - 1.) / self.pmask_slope
            # self.vmask.data = self.vmask.data.to(self.device)
            self.vmask = 0.1*torch.ones(self.image_dims, requires_grad=False).to(self.device)
        elif self.type == 'random':
            fmask = utils.get_random_mask(self.sparsity, self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).float().to(device)
        elif self.type == 'equispaced':
            fmask = utils.get_equispaced_mask(self.sparsity, self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).float().to(device)
        

    def forward(self):
        if self.type == 'learned':
            # Sample normal mask
            # std = torch.exp(0.5 * self.vmask)
            std = self.vmask

            eps = torch.randn_like(std)
            mask = self.mmask + eps*std
            # mask.register_hook(lambda grad: print('2', grad))
            mask = torch.nn.ReLU()(mask) + 1
            print(mask)
            # mask.register_hook(lambda grad: print('3', grad))
        else:
            mask = self.fmask

        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

class BernoulliFrameMask(nn.Module):
    def __init__(self, mask_type, image_dims, device, num_captures, sparsity, pmask_slope=5, eps=0.01, temp=0.8):
        super(BernoulliFrameMask, self).__init__()
        
        self.image_dims = image_dims
        self.type = mask_type
        self.device = device
        self.temp = temp
        self.sparsity = sparsity
        self.num_captures = num_captures
        self.sigmoid = nn.Sigmoid()

        if self.type == 'learned':
            self.pmask_slope = pmask_slope
            self.device = device
            
            # MaskNet
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            self.pmask.requires_grad = True
            self.pmask.data.uniform_(eps, 1-eps)
            self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
            self.pmask.data = self.pmask.data.to(self.device)
        elif self.type == 'random':
            fmask = utils.get_random_mask(self.sparsity, self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).float().to(device)
        elif self.type == 'equispaced':
            fmask = utils.get_equispaced_mask(self.sparsity, self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).float().to(device)
        
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, p, temperature):
        g1 = self.sample_gumbel(p.size())
        g2 = self.sample_gumbel(p.size())
        return 1-self.sigmoid((torch.log(1-p) - torch.log(p) + g1 - g2)/temperature)

    def binary_gumbel_softmax(self, pmask, temperature):
        """Shape-agnostic binary Gumbel-Softmax sampler

        input: (*) probabilistic mask
        return: (*) pixel-wise Bernoulli realization
        """
        y = self.gumbel_softmax_sample(pmask, temperature)
        y_hard = y.round()
        return (y_hard - y).detach() + y

    def squash_mask(self, mask):
        return self.sigmoid(self.pmask_slope*mask)

    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def forward(self):
        if self.type == 'learned':
            # Apply probabilistic mask
            mask = self.squash_mask(self.pmask)
            mask = self.sparsify(mask)
        else:
            mask = self.fmask

        mask = mask.unsqueeze(0).unsqueeze(0).repeat(self.num_captures, 1, 1, 1)
        # Sample
        mask = self.binary_gumbel_softmax(mask, self.temp)
        return mask
