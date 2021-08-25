"""
Model for CSFM.
For more details, please read:
    Alan Q. Wang, Aaron K. LaViolette, Leo Moon, Chris Xu, and Mert R. Sabuncu.
    "Joint Optimization of Hadamard Sensing and Reconstruction in Compressed Sensing Fluorescence Microscopy." 
    MICCAI 2021
"""
from . import utils, undersamplemask
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_measurement(noisy, mask):
    '''Generates under-sampled measurement in Hadamard space.'''
    _, _, nc, n1, n2 = noisy.shape

    y_noisy_frames = utils.hadamard_transform_torch(
        noisy.view(-1, nc, n1, n2)).view(noisy.shape)
    y_noisy_under = y_noisy_frames * mask

    y = torch.mean(y_noisy_under, dim=1)

    return y


class Upsample(nn.Module):
    """Upsample a multi-channel input image"""

    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Unet(nn.Module):
    def __init__(self, device, mask_type, imsize, sparsity, num_captures, nh, residual=True):
        super(Unet, self).__init__()

        self.device = device
        self.residual = residual
        self.sparsity = sparsity
        self.num_captures = num_captures

        self.dconv_down1 = self.double_conv(1, nh)
        self.dconv_down2 = self.double_conv(nh, nh)
        self.dconv_down3 = self.double_conv(nh, nh)
        self.dconv_down4 = self.double_conv(nh, nh)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = self.double_conv(nh+nh, nh)
        self.dconv_up2 = self.double_conv(nh+nh, nh)
        self.dconv_up1 = self.double_conv(nh+nh, nh)

        self.conv_last = nn.Conv2d(nh, 1, 1)

        self.bernoullimask = undersamplemask.BernoulliFrameMask(
            mask_type, (imsize, imsize), device, num_captures, sparsity)

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

    def forward(self, noisy_img):
        mask = self.bernoullimask()
        y = generate_measurement(noisy_img, mask)

        zf = utils.hadamard_transform_torch(y, normalize=False)

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
