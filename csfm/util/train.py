import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from csfm.util import utils
from csfm.model.model import Unet, HQSNet
from csfm.data import fmd

class BaseTrain(object):
  def __init__(self, args):
    self.conf = vars(args)

  def train(self):
    """Training loop. 

    Handles model, optimizer, loss, and sampler generation.
    Handles data loading. Handles i/o and checkpoint loading.
      

    Parameters
    ----------
    xdata : numpy.array (N, img_height, img_width, 2)
      Dataset of under-sampled measurements
    gt_data : numpy.array (N, img_height, img_width, 2)
      Dataset of fully-sampled images
    conf : dict
      Miscellaneous parameters

    Returns
    ----------
    network : regagcsmri.UNet
      Main network and hypernetwork
    optimizer : torch.optim.Adam
      Adam optimizer
    epoch_loss : float
      Loss for this epoch
    """
    ###############  Dataset ########################
    loader = fmd.load_denoising(self.conf['data_root'], train=True, 
      batch_size=self.conf['batch_size'], get_noisy=not self.conf['simulate'],
      types=None, captures=self.conf['captures'],
      transform=None, target_transform=None, 
      patch_size=self.conf['imsize'], test_fov=19)

    if self.conf['simulate']:
      val_loader = fmd.load_denoising_test_mix(self.conf['data_root'], 
        batch_size=self.conf['batch_size'], get_noisy=False, 
        transform=None, patch_size=self.conf['imsize'])
    else:
      val_loader = fmd.load_denoising(self.conf['data_root'], train=False, 
        batch_size=self.conf['batch_size'], get_noisy=True, 
        types=None, captures=self.conf['captures'],
        transform=None, target_transform=None, 
        patch_size=self.conf['imsize'], test_fov=19)
    ##################################################

    ##### Model, Optimizer, Loss ############
    if self.conf['model'] == 'unet':
      network = Unet(self.conf['device'], self.conf['mask_dist'], self.conf['mask_type'], \
          self.conf['imsize'], self.conf['accelrate'], self.conf['captures'], self.conf['unet_hidden']).to(self.conf['device'])
    else:
      network = HQSNet(K=5, mask=self.conf['mask'], lmbda=0, device=self.conf['device']).to(self.conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=self.conf['lr'])
    if self.conf['force_lr'] is not None:
      for param_group in optimizer.param_groups:
        param_group['lr'] = self.conf['force_lr']

    criterion = nn.MSELoss()
    ##################################################

    ############ Checkpoint Loading ##################
    if self.conf['load_checkpoint'] != 0:
      pretrain_path = os.path.join(self.conf['filename'], 'model.{epoch:04d}.h5'.format(epoch=self.conf['load_checkpoint']))
      network, optimizer = utils.load_checkpoint(network, pretrain_path, optimizer)
    ##################################################

    ############## Training loop #####################
    for epoch in range(self.conf['load_checkpoint']+1, self.conf['num_epochs']+1):
      print('\nEpoch %d/%d' % (epoch, self.conf['num_epochs']))
    
      # Train
      network, optimizer, train_epoch_loss, train_epoch_psnr = self.train_epoch(network, loader, criterion, optimizer)
      # Validate
      network, val_epoch_loss, val_epoch_psnr = self.eval_epoch(network, val_loader, criterion)
      # Save checkpoints
      print(train_epoch_psnr, val_epoch_psnr)
      utils.save_loss(epoch, train_epoch_loss, val_epoch_loss, self.conf['filename'])
      utils.save_loss(epoch, train_epoch_psnr, val_epoch_psnr, self.conf['filename'], 'psnr')
      if epoch % self.conf['log_interval'] == 0:
        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
            train_epoch_loss, val_epoch_loss, self.conf['filename'])

  def prepare_batch(self, datum):
    if self.conf['simulate']:
      clean_img, a, b = datum
      bs, ncrops, c, h, w = clean_img.shape
      clean_img = clean_img.float().to(self.conf['device']).view(-1, c, h, w)
      noisy_img = None
      a = a.float().to(self.conf['device']).view(-1)
      b = b.float().to(self.conf['device']).view(-1)
    else:
      noisy_img, clean_img = datum
      bs, frames, ncrops, c, h, w = noisy_img.shape
      noisy_img = noisy_img.permute(0, 2, 1, 3, 4, 5)
      noisy_img = noisy_img.float().to(self.conf['device']).view(-1, frames, c, h, w)
      clean_img = clean_img.float().to(self.conf['device']).view(-1, c, h, w)
      a = None
      b = None

    if self.conf['add_poisson_noise']:
      # Add Poisson noise
      noisy_img = torch.poisson(noisy_img + self.conf['poisson_const'])
    return noisy_img, clean_img, a, b

  def train_epoch(self, network, dataloader, criterion, optimizer):
    """Train for one epoch

      Parameters
      ----------
      network : UNet
        Main network and hypernetwork
      dataloader : torch.utils.data.DataLoader
        Training set dataloader
      optimizer : torch.optim.Adam
        Adam optimizer

      Returns
      ----------
      network : UNet
        Main network and hypernetwork
      optimizer : torch.optim.Adam
        Adam optimizer
      epoch_loss : float
        Loss for this epoch
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
      noisy_img, clean_img, a, b = self.prepare_batch(batch)
      
      optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        recon, mask = network(clean_img, noisy_img, self.conf['simulate'], a, b)
        clean_img = utils.normalize(clean_img)
        recon = utils.normalize(recon)
        if self.conf['mask_dist'] == 'normal':
          loss = (1-self.conf['lmbda'])*criterion(recon, clean_img) + self.conf['lmbda']*torch.norm(mask)
        else:
          loss = criterion(recon, clean_img)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.cpu().numpy()
        epoch_psnr += np.sum(utils.get_metrics(clean_img.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=False))
      epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, optimizer, epoch_loss, epoch_psnr

  def eval_epoch(self, network, dataloader, criterion):
    """Validate for one epoch

      Parameters
      ----------
      network : regagcsmri.UNet
        Main network and hypernetwork
      dataloader : torch.utils.data.DataLoader
        Training set dataloader
      hpsampler : regagcsmri.HpSampler
        Hyperparameter sampler
      conf : dict
        Miscellaneous parameters
      topK : int or None
        K for DHS sampling
      epoch : int
        Current training epoch

      Returns
      ----------
      network : regagcsmri.UNet
        Main network and hypernetwork
      epoch_loss : float
        Loss for this epoch
    """
    network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
      noisy_img, clean_img, a, b = self.prepare_batch(batch)
      with torch.set_grad_enabled(False):
        recon, mask = network(clean_img, noisy_img, self.conf['simulate'], a, b)
        clean_img = utils.normalize(clean_img)
        recon = utils.normalize(recon)
        if self.conf['mask_dist'] == 'normal':
          loss = (1-self.conf['lmbda'])*criterion(recon, clean_img) + self.conf['lmbda']*torch.norm(mask)
        else:
          loss = criterion(recon, clean_img)

        epoch_loss += loss.data.cpu().numpy()
        epoch_psnr += np.sum(utils.get_metrics(clean_img.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=False))
      epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss, epoch_psnr