import torch
import numpy as np
from csfm import utils, train, data, model
import argparse
import os
import time
from pprint import pprint
import json

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='CSFM')
        # data 
        self.add_argument('--data-root', type=str, default='/nfs02/users/aw847/data/denoising-fluorescent/', help='directory to dataset root')
        self.add_argument('--imsize', type=int, default=256)
        self.add_argument('--in-channels', type=int, default=1)
        self.add_argument('--out-channels', type=int, default=1)
        self.add_argument('--transform', type=str, default='four_crop', choices=['four_crop', 'center_crop'])
        self.add_argument('--noise-levels-train', type=list, default=[1])
        self.add_argument('--noise-levels-test', type=list, default=[1])
        self.add_argument('--captures', type=int, default=50, help='# captures per group')
        self.add_argument('--accelrate', type=float, default=0.25, help='# captures per group')
        self.add_argument('--model', type=str, choices=['unet', 'unroll'], help='arch of model')

        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='/nfs02/users/aw847/models/CSMPM/', type=str, help='directory to save models')
	
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Force learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
        self.add_argument('--log_interval', type=int, default=1, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--unet_hidden', type=int, default=64)
        self.add_argument('--temp', type=float, default=0.8)
        self.add_argument('--lmbda', type=float, default=0.5)

        self.add_argument('--force_date', type=str, default=None, help='Force learning rate')
        self.add_argument('--mask_type', type=str, choices=['learned', 'random', 'equispaced', 'uniform', 'halfhalf'], help='arch of model')
        self.add_argument('--mask_dist', type=str, choices=['normal', 'bernoulli'], help='arch of model')
        utils.add_bool_arg(self, 'simulate')

    def parse(self):
        args = self.parse_args()
        date = '{}'.format(time.strftime('%b_%d'))
        if args.force_date:
            date = args.force_date
        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date, \
            f'{args.model}_'\
            f'captures{args.captures}_'\
            f'bs{args.batch_size}_lr{args.lr}_'\
            f'accelrate{args.accelrate}_'f'temp{args.temp}_'f'masktype{args.mask_type}_'\
            f'dist{args.mask_dist}_'\
            f'lmbda{args.lmbda}_'\
            f'simulate{args.simulate}_'\
            f'nh{args.unet_hidden}')
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

if __name__ == "__main__":

    ############### Argument Parsing #################
    args = Parser().parse()
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    ##################################################

    trainer(vars(args))

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
        
        # plt.imshow(clean_img[0,0].cpu().detach().numpy())
        # plt.show()
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

