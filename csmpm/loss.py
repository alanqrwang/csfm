"""
Loss for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from . import utils

class AmortizedLoss(nn.Module):
    """Loss function for model"""
    def __init__(self, reg_types, range_restrict, mask, sampling_method, device, evaluate=False):
        """
        Parameters
        ----------
        reg_types : list
            List of strings which describes which regularization
                functions to use and in what order
        range_restrict : bool
            Whether or not to range restrict hyperparameter values
            if True, then for 1 and 2 hyperparameters, respectively:
                Loss = (1-alpha)*DC + alpha*Reg1
                Loss = alpha*DC + (1-alpha)*beta*Reg1 + (1-alpha)*(1-beta)*Reg2
            else:
                Loss = DC + alpha*Reg1
                Loss = DC + alpha*Reg1 + beta*Reg2
        mask : torch.Tensor 
            Undersampling mask
        sampling_method : string
            Either uhs or dhs
        evaluate : bool
            Train or val
        """
        super(AmortizedLoss, self).__init__()
        self.reg_types = reg_types
        self.range_restrict = range_restrict
        self.mask = mask
        self.sampling_method = sampling_method
        self.evaluate = evaluate
        self.device = device

    def get_tv(self, x):
        """Total variation loss

        Parameters
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Input image

        Returns
        ----------
        tv_loss : TV loss
        """
        x = x.permute(0, 3, 1, 2)
        tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs(), dim=(1, 2))
        return tv_x + tv_y
        
    def get_wavelets(self, x):
        """L1-penalty on wavelets

        Parameters
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Input image

        Returns
        ----------
        tv_loss : wavelets loss
        """
        xfm = DWTForward(J=3, mode='zero', wave='db4').to(self.device) # Accepts all wave types available to PyWavelets
        Yl, Yh = xfm(x)

        batch_size = x.shape[0]
        channels = x.shape[1]
        rows = utils.nextPowerOf2(Yh[0].shape[-2]*2)
        cols = utils.nextPowerOf2(Yh[0].shape[-1]*2)
        wavelets = torch.zeros(batch_size, channels, rows, cols).to(self.device)
        # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
        for i, band in enumerate(Yh):
            irow = rows // 2**(i+1)
            icol = cols // 2**(i+1)
            wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
            wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
            wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

        wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
        return wavelets

    def forward(self, x_hat, y, hyperparams, cap_reg, topK, schedule):
        '''
        Parameters
        ----------
        x_hat : torch.Tensor (batch_size, img_height, img_width, 2)
            Reconstructed image
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Matrix of hyperparameter values
        cap_reg : float
            Layer-wise L1 capacity penalty
        topK : int
            K for DHS sampling
        schedule : bool
            Loss scheduler

        Returns
        ----------
        loss : torch.Tensor (batch_size,)
            Total amortized loss per batch image
        loss_dict : dict
            Losses separated by different terms
        hyperparams : torch.Tensor
            Hyperparmeters used
        '''
        
        mask_expand = self.mask.unsqueeze(2)
        loss_dict = {}

        # Data consistency term
        l2 = torch.nn.MSELoss(reduction='none')
        Fx_hat = utils.fft(x_hat)
        UFx_hat = Fx_hat * mask_expand
        dc = torch.sum(l2(UFx_hat, y), dim=(1, 2, 3))
        loss_dict['dc'] = dc

        # Regularization
        loss_dict['cap'] = cap_reg
        tv = self.get_tv(x_hat)
        loss_dict['tv'] = tv
        wavelets = self.get_wavelets(x_hat)
        loss_dict['w'] = tv

        if len(self.reg_types) == 1:
            assert len(self.reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
            alpha = hyperparams[:, 0]
            if self.range_restrict:
                loss = (1-alpha)*dc + alpha*loss_dict[self.reg_types[0]]
            else:
                loss = dc + alpha*loss_dict[self.reg_types[0]]

        elif len(self.reg_types) == 2:
            assert len(self.reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
            alpha = hyperparams[:, 0]
            beta = hyperparams[:, 1]
            if self.range_restrict:
                if schedule:
                    loss = alpha * dc + (1-alpha)*beta * loss_dict[self.reg_types[0]] + (1-alpha)*(1-beta) * loss_dict[self.reg_types[1]]
                else:
                    loss = alpha * dc + (1-alpha) * loss_dict[self.reg_types[1]]
            else:
                loss = dc + alpha * loss_dict[self.reg_types[0]] + beta * loss_dict[self.reg_types[1]]

        else:
            assert len(self.reg_types) == hyperparams.shape[1] - 1, 'num_hyperparams and reg mismatch'
            loss = hyperparams[:,0]*loss_dict['dc']
            for i in range(len(self.reg_types)):
                loss += hyperparams[:,i+1] * loss_dict[self.reg_types[i]]
            loss = loss / torch.sum(hyperparams, dim=1)

        if self.evaluate:
            return loss, loss_dict, hyperparams
        else:
            loss, sort_hyperparams = self.process_losses(loss, dc, topK, hyperparams)
            return loss, loss_dict, sort_hyperparams

    def process_losses(self, unsup_losses, dc_losses, topK, hyperparams):
        '''Performs DHS sampling if topK is not None, otherwise UHS sampling

        Parameters
        ----------
        unsup_losses : torch.Tensor (batch_size,)
            Per-batch losses
        dc_losses : torch.Tensor (batch_size,)
            Per-batch DC losses
        topK : int
            K for DHS sampling
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Matrix of hyperparameter values

        Returns
        ----------
        loss : float
            Total amortized loss
        sort_hyperparams : torch.Tensor
            Hyperparmeters sorted by best DC
        '''
        if topK is not None:
            assert self.sampling_method == 'dhs'
            _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
            sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
            sort_hyperparams = hyperparams[perm]

            loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
        else:
            loss = torch.sum(unsup_losses)
            sort_hyperparams = hyperparams

        return loss, sort_hyperparams
