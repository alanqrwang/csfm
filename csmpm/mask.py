import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import layers, utils

class HadamardMask(nn.Module):
    def __init__(self, learned, image_dims, device, sparsity, pmask_slope=None, sample_slope=None, straight_through_mode=None, eps=0.01):
        super(HadamardMask, self).__init__()
        
        self.image_dims = image_dims
        self.learned = learned
        self.sparsity = sparsity
        if self.learned:
            self.pmask_slope = pmask_slope
            self.sample_slope = sample_slope
            self.device = device
            self.straight_through_mode = straight_through_mode
            self.sigmoid = nn.Sigmoid()
            
            # MaskNet
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            self.pmask.requires_grad = True
            self.pmask.data.uniform_(eps, 1-eps)
            self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
            self.pmask.data = self.pmask.data.to(self.device)
        else:
            # hmask = utils.get_mask(self.sparsity, self.image_dims)
            hmask = np.ones(self.image_dims)
            self.hmask = torch.tensor(hmask, requires_grad=False).float().to(device)
        
    def squash_mask(self, mask):
        return self.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty_like(self.pmask).uniform_(0, 1).to(self.device)
        return self.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self, epoch=0, tot_epochs=0):
        if self.learned:
            # Apply probabilistic mask
            probmask = self.squash_mask(self.pmask)
            # Sparsify
            sparse_mask = self.sparsify(probmask)
            # Threshold
            if self.straight_through_mode == 'ste-identity':
                stidentity = straight_through_sample.STIdentity.apply
                mask = stidentity(sparse_mask)
            elif self.straight_through_mode == 'ste-sigmoid':
                stsigmoid = straight_through_sample.STSigmoid.apply
                mask = stsigmoid(sparse_mask, epoch, tot_epochs)
            else:
                mask = self.threshold(sparse_mask)
        else:
            mask = self.hmask
        return mask
        
class FrameMask(nn.Module):
    def __init__(self, learned, image_dims, device, sparsity, pmask_slope=None, sample_slope=None, straight_through_mode=None, eps=0.01):
        super(FrameMask, self).__init__()
        
        self.image_dims = image_dims
        self.learned = learned
        self.sparsity = sparsity
        self.device = device

        if self.learned:
            self.pmask_slope = pmask_slope
            self.sample_slope = sample_slope
            self.device = device
            self.straight_through_mode = straight_through_mode
            self.sigmoid = nn.Sigmoid()
            
            # MaskNet
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            self.pmask.requires_grad = True
            self.pmask.data.uniform_(eps, 1-eps)
            self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
            self.pmask.data = self.pmask.data.to(self.device)
        else:
            fmask = utils.get_frame_mask(self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).long().to(device)
        
    def squash_mask(self, mask):
        return self.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        xbar = mask.mean()
        r = self.sparsity / xbar
        beta = (1-self.sparsity) / (1-xbar)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty_like(self.pmask).uniform_(0, 1).to(self.device)
        return self.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self, epoch=0, tot_epochs=0):
        if self.learned:
            # Apply probabilistic mask
            probmask = self.squash_mask(self.pmask)
            # Sparsify
            sparse_mask = self.sparsify(probmask)
            # Threshold
            if self.straight_through_mode == 'ste-identity':
                stidentity = straight_through_sample.STIdentity.apply
                mask = stidentity(sparse_mask)
            elif self.straight_through_mode == 'ste-sigmoid':
                stsigmoid = straight_through_sample.STSigmoid.apply
                mask = stsigmoid(sparse_mask, epoch, tot_epochs)
            else:
                mask = self.threshold(sparse_mask)
        else:
            mask = self.fmask
        return mask

class CategoricalFrameMask(nn.Module):
    '''Frame mask which outputs a one-hot representation of a realization of a categorical (discrete) random variable.

    Returns
    -------
    mask: (num_rows, num_cols, num_categories)
        E.g. mask[0, 0] is a one-hot vector of length num_categories
    '''
    def __init__(self, learned, image_dims, device, num_categories, eps=0.01, temp=0.8):
        super(CategoricalFrameMask, self).__init__()
        
        self.image_dims = image_dims
        self.learned = learned
        self.device = device
        self.temp = temp
        self.num_categories = num_categories

        if self.learned:
            # self.pmask_slope = pmask_slope
            # self.sample_slope = sample_slope
            self.device = device
            # self.straight_through_mode = straight_through_mode
            self.softmax = nn.Softmax(dim=1)
            
            # MaskNet
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0]*self.image_dims[1], num_categories))
            self.pmask.requires_grad = True
            self.pmask.data.uniform_(0, eps)
            # self.pmask.data = -torch.log(1. / self.pmask.data - 1.)
            # self.pmask.data = self.pmask.data.to(self.device)
            self.pmask.register_hook(lambda grad: print(grad))
        else:
            fmask = utils.get_frame_mask(self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).long().to(device)
        
    def squash_mask(self, mask):
        return self.softmax(mask)
    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward(self):
        if self.learned:
            # Apply probabilistic mask
            probmask = self.squash_mask(self.pmask)
            # Sample
            mask = self.gumbel_softmax(torch.log(probmask), self.temp).reshape(*self.image_dims, self.num_categories)
        else:
            mask = self.fmask
        return mask

class BernoulliFrameMask(nn.Module):
    '''Frame mask which outputs a realization of a Bernoulli random variable.

    Returns
    -------
    mask: (num_rows, num_cols)
        E.g. mask[0, 0] is a binary value {0, 1}
    '''

    def __init__(self, learned, image_dims, device, eps=0.01, temp=0.8):
        super(CategoricalFrameMask, self).__init__()
        
        self.image_dims = image_dims
        self.learned = learned
        self.device = device
        self.temp = temp

        if self.learned:
            self.pmask_slope = pmask_slope
            self.device = device
            # self.straight_through_mode = straight_through_mode
            self.sigmoid = nn.Sigmoid()
            
            # MaskNet
            self.pmask = nn.Parameter(torch.FloatTensor(self.image_dims[0], self.image_dims[1]))         
            self.pmask.requires_grad = True
            self.pmask.data.uniform_(eps, 1-eps)
            self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
            self.pmask.data = self.pmask.data.to(self.device)
            self.pmask.register_hook(lambda grad: print(grad))
        else:
            fmask = utils.get_frame_mask(self.image_dims)
            self.fmask = torch.tensor(fmask, requires_grad=False).long().to(device)
        
    def squash_mask(self, mask):
        return self.sigmoid(self.pmask_slope*mask)
    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def binary_gumbel_softmax(mask_p, temperature):
	"""
	input: (num_rows, num_cols) in [0, 1]
	return: (num_rows, num_cols) in {0, 1}
	"""
	p = mask_p.reshape(-1, 1)
	probs = torch.cat((p, 1-p), dim=1)
	logits = torch.log(probs)
	y = gumbel_softmax_sample(logits, temperature)
	_, ind = y.max(dim=-1)
	y_hard = ind
	return ((y_hard - y[:,0]).detach() + y[:,0]).view(*mask_p.shape)

    def forward(self):
        if self.learned:
            # Apply probabilistic mask
            probmask = self.squash_mask(self.pmask)
            # Sample
            mask = self.binary_gumbel_softmax(probmask, self.temp)
        else:
            mask = self.fmask
        return mask

class CondFrameMask(nn.Module):
    def __init__(self, image_dims, pmask_slope, sample_slope, sparsity, device, straight_through_mode, eps=0.01):
        super(CondMask, self).__init__()
        
        self.image_dims = image_dims
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        self.device = device
        self.sparsity = sparsity
        self.straight_through_mode = straight_through_mode
        self.sigmoid = nn.Sigmoid()
        
        # MaskNet outputs a vector of probabilities corresponding to image height
        self.fc1 = nn.Linear(1, self.image_dims[1])
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(self.image_dims[0], self.image_dims[0])

    def squash_mask(self, mask):
        # Takes in probability vector and outputs 2d probability mask  
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, self.image_dims[0], -1)
        return self.sigmoid(self.pmask_slope*mask)
    
    def sparsify(self, mask):
        mask_out = torch.zeros_like(mask)
        xbar = mask.mean(-1).mean(-1)
        r = self.sparsity / xbar
        r = r.view(-1, 1, 1)
        beta = (1-self.sparsity) / (1-xbar)
        beta = beta.view(-1, 1, 1)
        le = (r <= 1).float()
        return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

    def threshold(self, mask):
        random_uniform = torch.empty(mask.shape[0], self.image_dims[0]).uniform_(0, 1).to(self.device)
        random_uniform = random_uniform.unsqueeze(1)
        random_uniform = random_uniform.expand(-1, self.image_dims[0], -1)
        return self.sigmoid(self.sample_slope * (mask - random_uniform))
    
    def forward(self, condition, get_prob_mask=False, epoch=0, tot_epochs=0):
        fc_out = self.relu(self.fc1(condition))
        fc_out = self.fc_final(fc_out)

        # probmask is of shape (B, img_height)
        # Apply probabilistic mask
        probmask = self.squash_mask(fc_out)
        # Sparsify
        sparse_mask = self.sparsify(probmask)
        # Threshold
        if self.straight_through_mode == 'ste-identity':
            stidentity = straight_through_sample.STIdentity.apply
            mask = stidentity(sparse_mask)
        elif self.straight_through_mode == 'ste-sigmoid':
            stsigmoid = straight_through_sample.STSigmoid.apply
            mask = stsigmoid(sparse_mask, epoch, tot_epochs)
        else:
            mask = self.threshold(sparse_mask)

        return mask
