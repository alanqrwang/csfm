import torch
import torch.nn as nn
from . import layers, undersamplemask
from csfm.util import utils

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

    y_noisy_under = frames * mask

    y = torch.mean(y_noisy_under, dim=1)

  else:
    assert noisy is not None
    batch_size, tot_frames, nc, n1, n2 = noisy.shape

    y_noisy_frames = utils.hadamard_transform_torch(noisy.view(-1, nc, n1, n2)).view(noisy.shape)
    y_noisy_under = y_noisy_frames * mask

    y = torch.mean(y_noisy_under, dim=1)
  
  return y
class Unet(nn.Module):
  def __init__(self, device, mask_dist, mask_type, imsize, sparsity, num_captures, nh, residual=True):
    super(Unet, self).__init__()
      
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
    mask = self.bernoullimask()
    y = generate_measurement(clean_img, noisy_img, mask, simulate, a, b, self.num_captures)

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
  
    return out, mask