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
# import parse
import glob
from . import test
import myutils

def gen_hadamard_pattern(order, row_num, device):
    '''https://math.stackexchange.com/questions/1998761/how-to-get-the-value-of-hadamard-matrix-given-its-column-and-row-index
    '''
    num_bits = int(np.log2(order))
    nums = torch.arange(order)
    bin_nums = ((nums.reshape(-1,1) & (2**torch.arange(num_bits))) != 0).long()
    bin_row = ((torch.tensor(row_num).reshape(-1,1) & (2**torch.arange(num_bits))) != 0).long()
    exponents = torch.sum(bin_nums * bin_row.repeat(bin_nums.shape[0], 1), dim=1)
    return ((-1)**exponents).reshape(int(math.sqrt(order)), int(math.sqrt(order))).float().to(device)

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

def get_mask(sparsity, shape):
    n1, n2 = shape
    # undersample_rate = int(1/sparsity)
    # base = torch.zeros(undersample_rate)
    # base[0] = 1
    # mask = base.repeat(n1*n2 // undersample_rate)
    # mask = mask.reshape(n1, n2)
    # return mask

    # base = torch.zeros(n1*n2)
    # ones = torch.ones(n1*n2 // 2)
    # base[:n1*n2 // 2] = ones
    # mask = base.reshape(n1, n2)
    mask = torch.ones(shape)
    return mask

def get_frame_mask(shape):
    return torch.ones(shape)

def get_random_mask(undersample_rate, shape):
    n1, n2 = shape
    p = torch.empty(n1, n2).fill_(1/undersample_rate)
    mask = torch.bernoulli(p)
    return mask

def normalize(arr):
    """Normalizes a batch of images into range [0, 1]"""
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
        recons_pro = utils.normalize_recons(recons)
        gt_pro = utils.normalize_recons(gt)
    else:
        recons_pro = myutils.array.make_imshowable(recons)
        gt_pro = myutils.array.make_imshowable(gt)
    for i in range(len(recons)):
        metric = myutils.metrics.get_metric(recons_pro[i], gt_pro[i], metric_type)
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

def save_checkpoint(epoch, model_state, optimizer_state, loss, val_loss, model_folder, log_interval, scheduler=None):
    if epoch % log_interval == 0:
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