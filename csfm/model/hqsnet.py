import torch.nn as nn
from csfm.util import utils

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
