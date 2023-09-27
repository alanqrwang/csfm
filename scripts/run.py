import torch
import torch.nn as nn
from csfm import utils, data, model
import argparse
import os
import time
from pprint import pprint
import json
from tqdm import tqdm


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='CSFM')
        # data
        self.add_argument('--data-root', type=str,
                          default='denoising-fluorescent/', help='directory to dataset root')
        self.add_argument('--imsize', type=int, default=256)
        self.add_argument('--captures', type=int, default=50,
                          help='# captures per group')
        self.add_argument('--accelrate', type=float,
                          default=0.25, help='# captures per group')

        self.add_argument('-fp', '--filename_prefix', type=str,
                          help='filename prefix', required=True)
        self.add_argument('--models_dir', default='out/',
                          type=str, help='directory to save models')

        self.add_argument('--lr', type=float, default=1e-3,
                          help='Learning rate')
        self.add_argument('--batch_size', type=int,
                          default=8, help='Batch size')
        self.add_argument('--num_epochs', type=int, default=100,
                          help='Total training epochs')
        self.add_argument('--load_checkpoint', type=int, default=0,
                          help='Load checkpoint at specificed epoch')
        self.add_argument('--log_interval', type=int,
                          default=1, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0,
                          help='gpu id to train on')
        self.add_argument('--unet_hidden', type=int, default=64)

        self.add_argument('--mask_type', type=str, choices=[
                          'learned', 'random', 'equispaced', 'uniform', 'halfhalf'], help='arch of model')
        self.add_argument('--poisson_const', type=float, default=None,
                          help='Constant to add for Poisson noise')
        self.add_bool_arg('add_poisson_noise', default=False)

    def add_bool_arg(self, name, default=True):
        """Add boolean argument to argparse parser"""
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name: default})

    def validate_args(self, args):
        if args.add_poisson_noise:
          assert args.poisson_const is not None, 'Must set poisson constant'
          
    def parse(self):
        args = self.parse_args()
        self.validate_args(args)
        args.run_dir = os.path.join(args.models_dir, args.filename_prefix,
                                    f'captures{args.captures}_'
                                    f'bs{args.batch_size}_lr{args.lr}_'
                                    f'accelrate{args.accelrate}_'f'masktype{args.mask_type}_'
                                    f'nh{args.unet_hidden}_'
                                    f'std{args.poisson_const}'
                                    )
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')

        model_folder = args.ckpt_dir
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        args.filename = model_folder

        print('Arguments:')
        pprint(vars(args))

        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)

        return args


def trainer(conf):
    """Training loop."""
    # Dataset
    loader = data.load_denoising(conf['data_root'], train=True,
                                 batch_size=conf['batch_size'],
                                 types=None, captures=conf['captures'],
                                 transform=None, target_transform=None,
                                 patch_size=conf['imsize'], test_fov=19)

    val_loader = data.load_denoising(conf['data_root'], train=False,
                                     batch_size=conf['batch_size'],
                                     types=None, captures=conf['captures'],
                                     transform=None, target_transform=None,
                                     patch_size=conf['imsize'], test_fov=19)

    # Model, Optimizer, Loss
    network = model.Unet(conf['device'], conf['mask_type'],
                         conf['imsize'], conf['accelrate'], conf['captures'],
                         conf['unet_hidden']).to(conf['device'])
    optimizer = torch.optim.Adam(network.parameters(), lr=conf['lr'])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(conf['load_checkpoint']+1, conf['num_epochs']+1):
        print('\nEpoch %d/%d' % (epoch, conf['num_epochs']))

        network, optimizer, train_epoch_loss = train(
            network, loader, criterion, optimizer, conf)
        network, val_epoch_loss = eval(network, val_loader, criterion, conf)

        print('Train loss: {:04f}, Val loss: {:04f}'.format(
            train_epoch_loss, val_epoch_loss))

        utils.save_loss(epoch, train_epoch_loss,
                        val_epoch_loss, conf['filename'])
        if epoch % conf['log_interval'] == 0:
            utils.save_checkpoint(epoch, network.state_dict(), optimizer.state_dict(),
                                  train_epoch_loss, val_epoch_loss, conf['filename'])

def prepare_batch(datum, conf):
    noisy_img, clean_img = datum
    _, frames, _, c, h, w = noisy_img.shape
    noisy_img = noisy_img.permute(0, 2, 1, 3, 4, 5)
    noisy_img = noisy_img.float().to(
        conf['device']).reshape(-1, frames, c, h, w)
    clean_img = clean_img.float().to(conf['device']).reshape(-1, c, h, w)

    if conf['add_poisson_noise']:
      # Add Poisson noise
      noisy_img = torch.poisson(noisy_img + conf['poisson_const'])
    return noisy_img, clean_img

def train(network, dataloader, criterion, optimizer, conf):
    """Train for one epoch.

        network : model to train
        dataloader : training set dataloader
        criterion : loss function
        optimizer : optimizer to use
        conf : parameters
    """
    network.train()

    epoch_loss = 0
    epoch_samples = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        noisy_img, clean_img = prepare_batch(batch, conf)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            recon = network(noisy_img)
            clean_img = utils.normalize(clean_img)
            recon = utils.normalize(recon)
            loss = criterion(recon, clean_img)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    return network, optimizer, epoch_loss


def eval(network, dataloader, criterion, conf):
    """Validate for one epoch.

        network : network to validate
        dataloader : validation dataloader
        criterion : loss function
        conf : parameters
    """
    network.eval()

    epoch_loss = 0
    epoch_samples = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        noisy_img, clean_img = prepare_batch(batch, conf)
        with torch.set_grad_enabled(False):
            recon = network(noisy_img)
            clean_img = utils.normalize(clean_img)
            recon = utils.normalize(recon)
            loss = criterion(recon, clean_img)

            epoch_loss += loss.data.cpu().numpy()
        epoch_samples += len(clean_img)
    epoch_loss /= epoch_samples
    return network, epoch_loss


if __name__ == "__main__":
    args = Parser().parse()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    trainer(vars(args))
