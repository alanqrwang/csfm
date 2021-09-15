"""
Utility functions for CSMPM
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import numpy as np
import os
import pickle
import myutils
import math

def hadamard_transform_torch(img, normalize=True):
  """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
  n must be a power of 2.
  Parameters:
    u: Tensor of shape (..., n)
    normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
  Returns:
    product: Tensor of shape (..., n)
  """
  batch_size, nc, n1, n2 = img.shape
  u = img.reshape(batch_size, -1)
  m = int(np.log2(n1*n2))
  assert n1*n2 == 1 << m, 'n must be a power of 2'
  x = u[..., np.newaxis]
  for d in range(m)[::-1]:
    x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
  x = x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
  return x.reshape(batch_size, nc, n1, n2)

def h_plus(img, normalize=False):
  '''Unnormalized H^+'''
  b, nc, n1, n2 = img.shape
  if normalize:
    return 1/(2*math.sqrt(n1*n2)) * (torch.ones_like(img)*torch.sum(img) + hadamard_transform_torch(img, normalize=False))
  else:
    return 1/2 * (torch.ones_like(img)*torch.sum(img) + hadamard_transform_torch(img, normalize=False))
def h_minus(img, normalize=False):
  '''Unnormalized H^-'''
  b, nc, n1, n2 = img.shape
  if normalize:
    return 1/(2*math.sqrt(n1*n2)) * (torch.ones_like(img)*torch.sum(img) - hadamard_transform_torch(img, normalize=False))
  else:
    return 1/2 * (torch.ones_like(img)*torch.sum(img) - hadamard_transform_torch(img, normalize=False))

def get_uniform_mask(sparsity, shape):
  mask = torch.ones(shape) * sparsity
  return mask

def get_equispaced_mask(sparsity, shape):
  '''Binary mask with 0 and 1 spread evenly in the 2d mask'''
  n1, n2 = shape
  undersample_rate = int(1/sparsity)
  base = torch.zeros(undersample_rate)
  base[0] = 1
  mask = base.repeat(n1*n2 // undersample_rate)
  prob_mask = mask.reshape(n1, n2)
  return prob_mask

def get_random_mask(sparsity, shape):
  '''Binary mask with 0 and 1 sampled uniformly at random in the 2d mask'''
  p = torch.empty(shape).fill_(sparsity)
  prob_mask = torch.bernoulli(p)
  return prob_mask

def get_halfhalf_mask(sparsity):
  '''Binary mask with 0 and 1 sampled uniformly at random in the 2d mask'''
  if sparsity == 0.25:
    prob_mask = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/halfhalf_4_natural.npy')
  elif sparsity == 0.125:
    prob_mask = np.load('/nfs02/users/aw847/data/fluorescentmicroscopy/masks/halfhalf_8_natural.npy')
  else:
    raise Exception('No mask')
  return prob_mask

def normalize(arr):
  """Normalizes a batch of images into range [0, 1]"""
  if type(arr) is np.ndarray:
    if len(arr.shape) > 2:
      res = np.zeros(arr.shape)
      for i in range(len(arr)):
        res[i] = (arr[i] - np.min(arr[i])) / (np.max(arr[i]) - np.min(arr[i]))
      return res
    else:
      return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

  else:
    if len(arr.shape) > 2:
      res = torch.zeros_like(arr)
      for i in range(len(arr)):
        res[i] = (arr[i] - torch.min(arr[i])) / (torch.max(arr[i]) - torch.min(arr[i]))
      return res
    else:
      return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))

def get_metrics(gt, recons, metric_type, take_avg, normalized=True):
  metrics = []
  if normalized:
    recons_pro = normalize(recons)
    gt_pro = normalize(gt)
  else:
    recons_pro = myutils.array.make_imshowable(recons)
    gt_pro = myutils.array.make_imshowable(gt)

  if len(recons.shape) > 2:
    for i in range(len(recons)):
      metric = myutils.metrics.get_metric(recons_pro[i], gt_pro[i], metric_type)
      metrics.append(metric)
  else:
    metric = myutils.metrics.get_metric(recons_pro, gt_pro, metric_type)
    metrics.append(metric)

  if take_avg:
    return np.mean(np.array(metrics))
  else:
    return np.array(metrics)

######### Saving/Loading checkpoints ############
def load_checkpoint(model, path, optimizer=None, suppress=False):
  if not suppress:
    print('Loading checkpoint from', path)
  checkpoint = torch.load(path, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
  else:
    return model

def save_checkpoint(epoch, model_state, optimizer_state, loss, val_loss, model_folder, scheduler=None):
  state = {
    'epoch': epoch,
    'state_dict': model_state,
    'optimizer' : optimizer_state,
    'loss': loss,
    'val_loss': val_loss
  }
  if scheduler is not None:
    state['scheduler'] = scheduler.state_dict(),

  filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
  torch.save(state, filename.format(epoch=epoch))
  print('Saved checkpoint to', filename.format(epoch=epoch))

def save_loss(epoch, loss, val_loss, model_path, name=None):
  if name is None:
    pkl_path = os.path.join(model_path, 'losses.pkl')
  else:
    pkl_path = os.path.join(model_path, '%s.pkl' % name)
  if os.path.exists(pkl_path):
    f = open(pkl_path, 'rb') 
    losses = pickle.load(f)
    train_loss_list = losses['loss']
    val_loss_list = losses['val_loss']

    if epoch-1 < len(train_loss_list):
      train_loss_list[epoch-1] = loss
      val_loss_list[epoch-1] = val_loss
    else:
      train_loss_list.append(loss)
      val_loss_list.append(val_loss)

  else:
    train_loss_list = []
    val_loss_list = []
    train_loss_list.append(loss)
    val_loss_list.append(val_loss)

  loss_dict = {'loss' : train_loss_list, 'val_loss' : val_loss_list}
  f = open(pkl_path,"wb")
  pickle.dump(loss_dict,f)
  f.close()
  print('Saved loss to', pkl_path) 

def add_bool_arg(parser, name, default=True):
  """Add boolean argument to argparse parser"""
  group = parser.add_mutually_exclusive_group(required=False)
  group.add_argument('--' + name, dest=name, action='store_true')
  group.add_argument('--no_' + name, dest=name, action='store_false')
  parser.set_defaults(**{name:default})

def create_2d_sequency_mask(mask):
  '''2d natural to 2d sequency order'''
  seq = np.load('/share/sablab/nfs02/users/aw847/data/fluorescentmicroscopy/256x256_2d_seq_indices.npy').astype(int)
  reordered_mask = np.zeros(mask.shape)
  for i in range(mask.shape[1] * mask.shape[0]):
    coord = seq[i,:]
    reordered_mask[coord[0], coord[1]] = mask.flatten()[i]
  return reordered_mask

def create_2d_natural_mask(mask):
  '''2d sequency to 2d natural order'''
  seq = np.load('/share/sablab/nfs02/users/aw847/data/fluorescentmicroscopy/256x256_2d_seq_indices.npy').astype(int)
  seq_1d = seq[:,0] * 256 + seq[:,1]
  had_1d = np.argsort(seq_1d)
  had_2d = np.unravel_index(had_1d, (256, 256))
  reordered_mask = np.zeros(mask.shape)
  for i in range(mask.shape[1] * mask.shape[0]):
    reordered_mask[had_2d[0][i], had_2d[1][i]] = mask.flatten()[i]
  return reordered_mask
