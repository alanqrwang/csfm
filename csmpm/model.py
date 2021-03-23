"""
Model architecture for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import utils, undersamplemask
from . import layers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def generate_measurement(gt, noisy, mask, simulate, a, b, num_captures):
    '''Generates under-sampled measurement in Hadamard space

    If bernoulli, then simulate a fixed acquisition time and take
    num_captures realizations of the forward model. Then mask across
    realizations and average pixel-wise

    If normal, then mask directly encodes acquisition time
    '''
    if simulate:
        assert len(mask) == num_captures, 'Should be Bernoulli mask'
        assert a is not None and b is not None
        # plus = utils.h_plus(gt / num_captures, normalize=False)
        # minus = utils.h_minus(gt / num_captures, normalize=False)
        plus = utils.h_plus(gt, normalize=False)
        minus = utils.h_minus(gt, normalize=False)
        
        plus = plus.unsqueeze(1).repeat(1, num_captures, 1, 1, 1)
        minus = minus.unsqueeze(1).repeat(1, num_captures, 1, 1, 1)
        
        a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        a_inv = torch.reciprocal(a)

        noisy_plus = torch.poisson(a_inv * plus)
        noisy_minus = torch.poisson(a_inv * minus)

        frames = a*noisy_plus - a*noisy_minus + torch.normal(0, std=b) - torch.normal(0, std=b)
        # frame1 = utils.create_2d_sequency_mask(torch.log(frames[0,0, 0]))
        # plt.imshow(frame1)
        # plt.show()
        # frame2 = utils.create_2d_sequency_mask(torch.log(frames[0,1, 0]))
        # plt.imshow(frame2)
        # plt.show()
        # frame3 = utils.create_2d_sequency_mask(torch.log(frames[0,2, 0]))
        # plt.imshow(frame3)
        # plt.show()

        y_noisy_under = frames * mask

        y = torch.mean(y_noisy_under, dim=1)

    else:
        assert noisy is not None
        batch_size, tot_frames, nc, n1, n2 = noisy.shape

        y_noisy_frames = utils.hadamard_transform_torch(noisy.view(-1, nc, n1, n2)).view(noisy.shape)
        y_noisy_under = y_noisy_frames * mask
        # plt.imshow(noisy[0,0, 0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(noisy[0,1,0].cpu().detach().numpy())
        # plt.show()
        # plt.imshow(noisy[0,2,0].cpu().detach().numpy())
        # plt.show()

        y = torch.mean(y_noisy_under, dim=1)
        
    return y

class BaseUnet(nn.Module):
    def __init__(self, device, mask_dist, mask_type, imsize, sparsity, num_captures, nh, residual=True, legacy=False):
        super(BaseUnet, self).__init__()
                
        self.device = device
        self.residual = residual
        self.sparsity = sparsity
        self.num_captures = num_captures
        self.mask_dist = mask_dist

        self.dconv_down1 = self.double_conv(1, nh)
        self.dconv_down2 = self.double_conv(nh, nh)
        self.dconv_down3 = self.double_conv(nh, nh)
        self.dconv_down4 = self.double_conv(nh, nh)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(nh+nh, nh)
        self.dconv_up2 = self.double_conv(nh+nh, nh)
        self.dconv_up1 = self.double_conv(nh+nh, nh)
        
        self.conv_last = nn.Conv2d(nh, 1, 1)

        if mask_dist == 'bernoulli':
            self.bernoullimask = undersamplemask.BernoulliFrameMask(mask_type, (imsize, imsize), device, num_captures, sparsity)
        elif mask_dist == 'normal':
            self.bernoullimask = undersamplemask.NormalMask(mask_type, (imsize, imsize), device)


    def double_conv(self, in_channels, out_channels, use_batchnorm=False):
        if use_batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )   
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )   
        

    def forward(self, clean_img, noisy_img, simulate, a=None, b=None):
        # plt.imshow(clean_img[0,0].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        mask = self.bernoullimask()
        # mask.register_hook(lambda grad: print('mask', grad))
        y = generate_measurement(clean_img, noisy_img, mask, simulate, a, b, self.num_captures)
        # y.register_hook(lambda grad: print('gen measurement', grad))

        zf = utils.hadamard_transform_torch(y, normalize=False)
        # zf.register_hook(lambda grad: print('inverse h', grad))
        # plt.imshow(zf[0,0].cpu().detach().numpy(), cmap='gray')
        # plt.show()
        # zf = utils.normalize(zf)
        # zf.register_hook(lambda grad: print('zf normalize', grad))
        # plt.imshow(zf[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        conv1 = self.dconv_down1(zf)
        # conv1.register_hook(lambda grad: print('conv1', grad))
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
        
        return out, mask

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
