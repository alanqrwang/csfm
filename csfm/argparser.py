import argparse
from pprint import pprint
import os
import time
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
        self.add_argument('--model', type=str, default='unet', choices=['unet', 'unroll'], help='arch of model')

        self.add_argument('-fp', '--filename_prefix', type=str, help='filename prefix', required=True)
        self.add_argument('--models_dir', default='/nfs02/users/aw847/models/CSMPM/', type=str, help='directory to save models')
	
        self.add_argument('--date', type=str, required=True, help='Date')
        self.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
        self.add_argument('--force_lr', type=float, default=None, help='Force learning rate')
        self.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.add_argument('--num_epochs', type=int, default=100, help='Total training epochs')
        self.add_argument('--load_checkpoint', type=int, default=0, help='Load checkpoint at specificed epoch')
        self.add_argument('--log_interval', type=int, default=25, help='Frequency of logs')
        self.add_argument('--gpu_id', type=int, default=0, help='gpu id to train on')
        self.add_argument('--unet_hidden', type=int, default=64)
        self.add_argument('--temp', type=float, default=0.8)
        self.add_argument('--lmbda', type=float, default=0.5)

        self.add_argument('--force_date', type=str, default=None, help='Force learning rate')
        self.add_argument('--mask_type', type=str, default='learned', choices=['learned', 'random', 'equispaced', 'uniform', 'halfhalf'], help='arch of model')
        self.add_argument('--mask_dist', type=str, default='bernoulli', choices=['normal', 'bernoulli'], help='arch of model')
        self.add_argument('--method', type=str, required=True, choices=['raw'], help='arch of model')
        self.add_bool_arg('simulate', default=False)
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
        date = args.date if args.date is not None else '{}'.format(time.strftime('%b_%d'))
        args.run_dir = os.path.join(args.models_dir, args.filename_prefix, date,
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