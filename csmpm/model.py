"""
Model architecture for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import utils, undersamplemask
from . import loss as losslayer
from . import layers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class BaseUnet(nn.Module):
    def __init__(self, device, mask_type, imsize, sparsity, num_captures, residual=True, legacy=False):
        super(BaseUnet, self).__init__()
                
        self.device = device
        self.residual = residual
        self.dconv_down1 = self.double_conv(1, 64)
        self.dconv_down2 = self.double_conv(64, 64)
        self.dconv_down3 = self.double_conv(64, 64)
        self.dconv_down4 = self.double_conv(64, 64)        
        self.sparsity = sparsity

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(128, 64)
        self.dconv_up2 = self.double_conv(128, 64)
        self.dconv_up1 = self.double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)

        if legacy:
            self.fmask = undersamplemask.BernoulliFrameMask(mask_type, (imsize, imsize), device, num_captures, sparsity)
        else:
            self.bernoullimask = undersamplemask.BernoulliFrameMask(mask_type, (imsize, imsize), device, num_captures, sparsity)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )   
        
    def generate_measurement(self, frames, mask):
        '''Generates under-sampled and under-framed measurement in Hadamard space

        mask: (num_frames, n1, n2) Realization of Bernoulli random variable
        frames: (batch_size, num_frames, nc, n1, n2) Stack of frames of single FOV
        '''
        batch_size, tot_frames, nc, n1, n2 = frames.shape

        # divisor = torch.cumsum(torch.ones_like(frames), dim=1)
        # avg_frames = frames.cumsum(dim=1) / divisor # Cumulative average of all frames, along dim=1
        # y_avg_frames = utils.hadamard_transform_torch(avg_frames) # Noisy -> Clean images in Hadamard basis, along dim=1
        y_noisy_frames = utils.hadamard_transform_torch(frames.view(-1, nc, n1, n2)).view(frames.shape)
        y_noisy_under = y_noisy_frames * mask

        avg_y = torch.mean(y_noisy_under, dim=1)

        return avg_y

    def forward(self, noisy_img):
        mask = self.bernoullimask()
        y = self.generate_measurement(noisy_img, mask)

        zf = utils.hadamard_transform_torch(y)

        conv1 = self.dconv_down1(zf)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        if self.residual:
            out = zf + out 
        
        return out

class ResBlock(nn.Module):
    '''5-layer CNN with residual output'''
    def __init__(self, n_ch_in=1, n_ch_out=1, nf=64, ks=3):
        '''
        Parameters
        ----------
        n_ch_in : int
            Number of input channels
        n_ch_out : int
            Number of output channels
        nf : int
            Number of hidden channels
        ks : int
            Kernel size
        '''
        super(ResBlock, self).__init__()
        self.n_ch_out = n_ch_out

        self.conv1 = nn.Conv2d(n_ch_in, nf, ks, padding = ks//2)
        self.conv2 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv5 = nn.Conv2d(nf, n_ch_out, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out = self.relu(conv3_out)

        conv4_out = self.conv4(conv3_out)
        conv4_out = self.relu(conv4_out)

        conv5_out = self.conv5(conv4_out)

        x_res = x[:,:self.n_ch_out,:,:] + conv5_out
        return x_res

class HQSNet(nn.Module):
    """HQSNet model architecture"""
    def __init__(self, K, mask, lmbda, device, n_hidden=64):
        """
        Parameters
        ----------
        K : int
            Number of unrolled iterations
        mask : torch.Tensor (img_height, img_width)
            Under-sampling mask
        lmbda : float
            Lambda value
        device : str
            Pytorch device string
        n_hidden : int
            Number of hidden dimensions
        """
        super(HQSNet, self).__init__()

        self.mask = mask
        self.lmbda = lmbda
        self.resblocks = nn.ModuleList()
        self.device = device
            
        for i in range(K):
            resblock = ResBlock(n_ch_in=1, nf=n_hidden)
            self.resblocks.append(resblock)

        self.block_final = ResBlock(n_ch_in=1, nf=n_hidden)

    def data_consistency(self, k, k0, mask, order, lmbda=0):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        order - order of Hadamard matrix
        """
        mask = mask.unsqueeze(0)
        mask = mask.expand_as(k)

        # return (1 - mask) * k + mask * (lmbda*k + k0) / (2**order + lmbda)
        return (1 - mask) * k + mask * (lmbda*k + k0) / (1 + lmbda)

    def forward(self, y):
        """
        Parameters
        ----------
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurement in Hadamard space
        """
        x = utils.hadamard_transform_torch(y)
        for i in range(len(self.resblocks)):
            # z-minimization
            z = self.resblocks[i](x)
            
            # x-minimization
            z_ksp = utils.hadamard_transform_torch(z)
            x_ksp = self.data_consistency(z_ksp, y, self.mask, order=y.shape[1]*y.shape[2], lmbda=self.lmbda)
            x = utils.hadamard_transform_torch(x_ksp)

        x = self.block_final(x)
        return x
