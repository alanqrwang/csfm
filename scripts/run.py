import torch
import numpy as np
from csmpm import utils, train, data, model
import argparse
import os
import time
from pprint import pprint
import json

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='DnCNN')
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

    train.trainer(vars(args))
