import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from csfm.util import utils
from csfm.model.unet import Unet
from csfm.model.hqsnet import HQSNet
from csfm.data import fmd

class BaseTrain(object):
  def __init__(self, args):
    self.device = args.device
    self.seed = args.seed
    self.run_dir = args.run_dir
    self.data_dir = args.data_dir
    self.imsize = args.imsize
    self.in_channels = args.in_channels
    self.out_channels = args.out_channels
    self.transform = args.transform
    self.noise_levels_train = args.noise_levels_train
    self.noise_levels_test = args.noise_levels_test
    self.captures = args.captures
    self.accelrate = args.accelrate
    self.arch = args.arch
    self.date = args.date
    self.lr = args.lr
    self.batch_size = args.batch_size
    self.num_epochs = args.num_epochs
    self.load_checkpoint = args.load_checkpoint
    self.log_interval = args.log_interval
    self.gpu_id = args.gpu_id
    self.unet_hidden = args.unet_hidden
    self.temp = args.temp
    self.lmbda = args.lmbda

    self.mask_type = args.mask_type
    self.mask_dist = args.mask_dist
    self.method = args.method
    self.simulate = args.simulate
    self.poisson_const = args.poisson_const
    self.add_poisson_noise = args.add_poisson_noise

  def set_random_seed(self):
    seed = self.seed
    if seed > 0:
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)

  def config(self):
    self.set_random_seed()

    self.get_dataloader()
    self.network = self.get_model()
    self.optimizer = self.get_optimizer()
    self.scheduler = self.get_scheduler()
    self.criterion = self.get_criterion()

  def get_model(self):
    if self.arch == 'unet':
      return Unet(self.device, self.mask_dist, self.mask_type, \
          self.imsize, self.accelrate, self.captures, self.unet_hidden).to(self.device)
    else:
      return HQSNet(K=5, mask=self.mask, lmbda=0, device=self.device).to(self.device)
  
  def get_optimizer(self):
    return torch.optim.Adam(self.network.parameters(), lr=self.lr)
  
  def get_scheduler(self):
    return None
  
  def get_criterion(self):
    return nn.MSELoss()

  def get_dataloader(self):
    self.train_loader = fmd.load_denoising(self.data_dir, train=True, 
      batch_size=self.batch_size, get_noisy=not self.simulate,
      types=None, captures=self.captures,
      transform=None, target_transform=None, 
      patch_size=self.imsize, test_fov=19)

    if self.simulate:
      self.val_loader = fmd.load_denoising_test_mix(self.data_dir, 
        batch_size=self.batch_size, get_noisy=False, 
        transform=None, patch_size=self.imsize)
    else:
      self.val_loader = fmd.load_denoising(self.data_dir, train=False, 
        batch_size=self.batch_size, get_noisy=True, 
        types=None, captures=self.captures,
        transform=None, target_transform=None, 
        patch_size=self.imsize, test_fov=19)

  def train_begin(self):
    # Directories to save information
    self.ckpt_dir = os.path.join(self.run_dir, 'checkpoints')
    if not os.path.exists(self.ckpt_dir):
      os.makedirs(self.ckpt_dir)

  def train(self):
    '''Training loop.'''

    ############ Checkpoint Loading ##################
    if self.load_checkpoint != 0:
      pretrain_path = os.path.join(self.filename, 'model.{epoch:04d}.h5'.format(epoch=self.load_checkpoint))
      self.network, self.optimizer = utils.load_checkpoint(self.network, pretrain_path, self.optimizer)
    ##################################################

    self.train_begin()
    for epoch in range(self.load_checkpoint+1, self.num_epochs+1):
      print('\nEpoch %d/%d' % (epoch, self.num_epochs))
    
      train_epoch_loss, train_epoch_psnr = self.train_epoch()
      val_epoch_loss, val_epoch_psnr = self.eval_epoch()
      print("train loss={:.6f}, train psnr={:.6f}".format(train_epoch_loss, train_epoch_psnr))
      print("val loss={:.6f}, val psnr={:.6f}".format(val_epoch_loss, val_epoch_psnr))

      # Save checkpoints
      utils.save_loss(epoch, train_epoch_loss, val_epoch_loss, self.run_dir)
      utils.save_loss(epoch, train_epoch_psnr, val_epoch_psnr, self.run_dir, 'psnr')
      if epoch % self.log_interval == 0:
        utils.save_checkpoint(epoch, self.network.state_dict(), self.optimizer.state_dict(), \
            train_epoch_loss, val_epoch_loss, self.ckpt_dir)

  def prepare_batch(self, datum):
    if self.simulate:
      clean_img, a, b = datum
      bs, ncrops, c, h, w = clean_img.shape
      clean_img = clean_img.float().to(self.device).view(-1, c, h, w)
      noisy_img = None
      a = a.float().to(self.device).view(-1)
      b = b.float().to(self.device).view(-1)
    else:
      noisy_img, clean_img = datum
      bs, frames, ncrops, c, h, w = noisy_img.shape
      noisy_img = noisy_img.permute(0, 2, 1, 3, 4, 5)
      noisy_img = noisy_img.float().to(self.device).view(-1, frames, c, h, w)
      clean_img = clean_img.float().to(self.device).view(-1, c, h, w)
      a = None
      b = None

    if self.add_poisson_noise:
      # Add Poisson noise
      noisy_img = torch.poisson(noisy_img + self.poisson_const)
    return noisy_img, clean_img, a, b

  def train_epoch(self):
    '''Train for one epoch'''
    self.network.train()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch in tqdm(self.train_loader, total=len(self.train_loader)):
      noisy_img, clean_img, a, b = self.prepare_batch(batch)
      
      self.optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        recon, mask = self.network(clean_img, noisy_img, self.simulate, a, b)
        clean_img = utils.normalize(clean_img)
        recon = utils.normalize(recon)
        if self.mask_dist == 'normal':
          loss = (1-self.lmbda)*self.criterion(recon, clean_img) + self.lmbda*torch.norm(mask)
        else:
          loss = self.criterion(recon, clean_img)

        loss.backward()
        self.optimizer.step()

        epoch_loss += loss.data.cpu().numpy()
        epoch_psnr += np.sum(utils.get_metrics(clean_img.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=False))
      epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return epoch_loss, epoch_psnr

  def eval_epoch(self):
    '''Validate for one epoch'''
    self.network.eval()

    epoch_loss = 0
    epoch_samples = 0
    epoch_psnr = 0

    for batch in tqdm(self.val_loader, total=len(self.val_loader)):
      noisy_img, clean_img, a, b = self.prepare_batch(batch)
      with torch.set_grad_enabled(False):
        recon, mask = self.network(clean_img, noisy_img, self.simulate, a, b)
        clean_img = utils.normalize(clean_img)
        recon = utils.normalize(recon)
        if self.mask_dist == 'normal':
          loss = (1-self.lmbda)*self.criterion(recon, clean_img) + self.lmbda*torch.norm(mask)
        else:
          loss = self.criterion(recon, clean_img)

        epoch_loss += loss.data.cpu().numpy()
        epoch_psnr += np.sum(utils.get_metrics(clean_img.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=False))
      epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return epoch_loss, epoch_psnr