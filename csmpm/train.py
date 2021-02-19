"""
Training loop for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import utils, model, data
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sys
import glob
import os
import math
import matplotlib.pyplot as plt

def trainer(conf):
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
    loader = data.load_denoising(conf['data_root'], train=True, 
        batch_size=conf['batch_size'], get_noisy=not conf['simulate'],
        types=None, captures=conf['captures'],
        transform=None, target_transform=None, 
        patch_size=conf['imsize'], test_fov=19)

    if conf['simulate']:
        val_loader = data.load_denoising_test_mix(conf['data_root'], 
            batch_size=conf['batch_size'], get_noisy=False, 
            transform=None, patch_size=conf['imsize'])
    else:
        val_loader = data.load_denoising(conf['data_root'], train=False, 
            batch_size=conf['batch_size'], get_noisy=True, 
            types=None, captures=conf['captures'],
            transform=None, target_transform=None, 
            patch_size=conf['imsize'], test_fov=19)
    ##################################################

    ##### Model, Optimizer, Loss ############
    if conf['model'] == 'unet':
        network = model.BaseUnet(conf['device'], conf['mask_dist'], conf['mask_type'], \
                conf['imsize'], conf['accelrate'], conf['captures'], conf['unet_hidden']).to(conf['device'])
    else:
        network = model.HQSNet(K=5, mask=conf['mask'], lmbda=0, device=conf['device']).to(conf['device'])

    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    if conf['force_lr'] is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = conf['force_lr']

    criterion = nn.MSELoss()
    ##################################################

    ############ Checkpoint Loading ##################
    if conf['load_checkpoint'] != 0:
        pretrain_path = os.path.join(conf['filename'], 'model.{epoch:04d}.h5'.format(epoch=conf['load_checkpoint']))
        network, optimizer = utils.load_checkpoint(network, pretrain_path, optimizer)
    ##################################################

    ############## Training loop #####################
    for epoch in range(conf['load_checkpoint']+1, conf['epochs']+1):
        print('\nEpoch %d/%d' % (epoch, conf['epochs']))
    
        # Train
        network, optimizer, train_epoch_loss, train_epoch_psnr = train(network, loader, criterion, optimizer, conf)
        # Validate
        network, val_epoch_loss, val_epoch_psnr = validate(network, val_loader, criterion, conf)
        # Save checkpoints
        # print(train_epoch_psnr, val_epoch_psnr)
        utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(), \
                train_epoch_loss, val_epoch_loss, conf['filename'], conf['log_interval'])
        utils.save_loss(epoch, train_epoch_loss, val_epoch_loss, conf['filename'])
        utils.save_loss(epoch, train_epoch_psnr, val_epoch_psnr, conf['filename'], 'psnr')

def train(network, dataloader, criterion, optimizer, conf):
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

    for batch_idx, datum in tqdm(enumerate(dataloader), total=len(dataloader)):
        if conf['simulate']:
            clean_img, a, b = datum
            bs, ncrops, c, h, w = clean_img.shape
            clean_img = clean_img.float().to(conf['device']).view(-1, c, h, w)
            noisy_img = None
            a = a.float().to(conf['device']).view(-1)
            b = b.float().to(conf['device']).view(-1)
        else:
            noisy_img, clean_img = datum
            bs, frames, ncrops, c, h, w = noisy_img.shape
            noisy_img = noisy_img.permute(0, 2, 1, 3, 4, 5)
            noisy_img = noisy_img.float().to(conf['device']).view(-1, frames, c, h, w)
            clean_img = clean_img.float().to(conf['device']).view(-1, c, h, w)
            a = None
            b = None
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            recon, mask = network(clean_img, noisy_img, conf['simulate'], a, b)
            clean_img = utils.normalize(clean_img)
            recon = utils.normalize(recon)
            if conf['mask_dist'] == 'normal':
                loss = (1-conf['lmbda'])*criterion(recon, clean_img) + conf['lmbda']*torch.norm(mask)
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

def validate(network, dataloader, criterion, conf):
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

    for batch_idx, datum in tqdm(enumerate(dataloader), total=len(dataloader)):
        if conf['simulate']:
            clean_img, a, b = datum
            bs, ncrops, c, h, w = clean_img.shape
            clean_img = clean_img.float().to(conf['device']).view(-1, c, h, w)
            noisy_img = None
            a = a.float().to(conf['device']).view(-1)
            b = b.float().to(conf['device']).view(-1)
        else:
            noisy_img, clean_img = datum
            print(noisy_img.shape)
            bs, frames, ncrops, c, h, w = noisy_img.shape
            noisy_img = noisy_img.permute(0, 2, 1, 3, 4, 5)
            noisy_img = noisy_img.float().to(conf['device']).view(-1, frames, c, h, w)
            clean_img = clean_img.float().to(conf['device']).view(-1, c, h, w)
            a = None
            b = None
        with torch.set_grad_enabled(False):
            recon, mask = network(clean_img, noisy_img, conf['simulate'], a, b)
            clean_img = utils.normalize(clean_img)
            recon = utils.normalize(recon)
            if conf['mask_dist'] == 'normal':
                loss = (1-conf['lmbda'])*criterion(recon, clean_img) + conf['lmbda']*torch.norm(mask)
            else:
                loss = criterion(recon, clean_img)

            epoch_loss += loss.data.cpu().numpy()
            epoch_psnr += np.sum(utils.get_metrics(clean_img.permute(0, 2, 3, 1), recon.permute(0, 2, 3, 1), 'psnr', False, normalized=False))
        epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    epoch_psnr /= epoch_samples
    return network, epoch_loss, epoch_psnr

