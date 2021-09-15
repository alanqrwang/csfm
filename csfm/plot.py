import os
import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from csfm.model import Unet
from csfm import utils
import torch

def plotloss(model_dirs, labels=None, ylim=None, xlim=None):
  if not isinstance(model_dirs, (tuple, list)):
    model_dirs = [model_dirs]
  if not isinstance(labels, (tuple, list)):
    labels = [labels]
  
  if labels[0] is None:
    labels = ['Line %d' % n for n in range(len(model_dirs))]
  assert len(labels) == len(model_dirs), 'labels do not match model paths'

  fig, axes = plt.subplots(1, 1)
  fig.set_size_inches(18.5, 10.5)
  for i, model_dir in enumerate(model_dirs):
    pkl_path = os.path.join(model_dir, 'checkpoints', 'losses.pkl')
    with (open(pkl_path, "rb")) as openfile:
      loss_dict = pickle.load(openfile)
    loss = loss_dict['loss']
    val_loss = loss_dict['val_loss']
 
    color = next(axes._get_lines.prop_cycler)['color']
    xvalues = np.arange(1, len(loss)+1) 
    axes.plot(xvalues, loss, color=color, label=labels[i])
    axes.plot(xvalues, val_loss, color=color, linestyle='dashed')
 
    if ylim is not None:
        axes.set_ylim(ylim)
    if xlim is not None:
        axes.set_xlim(xlim)
    axes.grid()
    axes.legend()

def plot_img(img, title=None, ax=None, rot90=False, ylabel=None, xlabel=None, vlim=None, colorbar=False):
  ax = ax or plt.gca()
  if rot90:
    img = np.rot90(img, k=1)

  if vlim:
    im = ax.imshow(img, vmin=vlim[0], vmax=vlim[1], cmap='gray')
  else:
    im = ax.imshow(img, cmap='gray')
  if title is not None:
    ax.set_title(title, fontsize=16)
  ax.set_xticks([])
  ax.set_yticks([])
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
  plt.colorbar(im, ax=ax)

  return ax, im

def get_learned_mask(ckpt_path, accelrate, mask_type, device, nh=64):
    print(ckpt_path)
    models = [sorted(glob(ckpt_path))[-1]]
#     models = sorted(glob.glob(ckpt_path))
    for i, model_path in enumerate(models[::20]):
        network = Unet(device, mask_type, 256, accelrate, 50, nh=nh).to(device) 
        network = utils.load_checkpoint(network, model_path, suppress=True)
        network.eval()
        
        pmask = network.bernoullimask.sparsify(network.bernoullimask.squash_mask(network.bernoullimask.pmask.data))
        fmask = network.bernoullimask()
        # Sum up Bernoulli realizations along the frame dimension
        fmask = torch.sum(fmask, dim=0)[0]
#         fmask_photons = photons_per_pixel(fmask)
        h_numpy = pmask.cpu().detach().numpy()#[100:110, 100:110]
        f_numpy = fmask.cpu().detach().numpy()#[100:110, 100:110]
#         f_photon_numpy = fmask_photons.cpu().detach().numpy()
        h_numpy_seq = utils.create_2d_sequency_mask(h_numpy)
        f_numpy_seq = utils.create_2d_sequency_mask(f_numpy)

        # Photons per pixel
#         fig, axes = plt.subplots(1, 2, figsize=(10, 6))
#         myutils.plot.plot_img(f_photon_numpy, ax=axes[0], vlim=[0, 0.5], colorbar=True, title='Photons per pixel')
#         axes[1].hist(f_photon_numpy.flatten(), bins=21, range=(0, 0.5))
#         plt.show()
        # Histograms of Mask
        # fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        # axes[0].hist(h_numpy.flatten())
        # axes[1].hist(f_numpy.flatten())
        # plt.show()
            
    return h_numpy, f_numpy, h_numpy_seq, f_numpy_seq

def plot_histogram(data, title=None, ylim=None, ax=None):
  ax = ax or plt.gca()
  ax.hist(data.flatten())
  if title:
    ax.set_title(title)
  if ylim:
    ax.set_ylim(ylim)
  return ax