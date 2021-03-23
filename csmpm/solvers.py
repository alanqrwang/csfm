import torch
import torch.nn as nn
from . import model, utils
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
from torch_radon.shearlet import ShearletTransform

class ImageParameter(nn.Module):
    def __init__(self, image_dims, init):
        super(ImageParameter, self).__init__()
        self.image_dims = image_dims

        self.x = nn.Parameter(torch.FloatTensor(image_dims))
        self.x.requires_grad = True
        self.x.data = init
        # self.x.register_hook(lambda grad: print(grad))

    def forward(self):
        return self.x

def gradient_descent(y, mask, until_convergence, max_iters, tol, lmbda, lr, device, shearlet):
    '''Gradient descent optimization of regularized regression with TV penalty'''
    loss_list = []
    prev_loss = torch.tensor(0.).float().to(device)
    reg_loss = torch.tensor(1e10).float().to(device)
    iters = 0

    # y = utils.normalize(y)
    zf = utils.hadamard_transform_torch(y, normalize=True)
    zf_nonorm = utils.hadamard_transform_torch(y, normalize=False)

    # plt.imshow(zf[0,0].cpu().detach().numpy())
    # plt.show()
    # plt.imshow(zf_nonorm[0,0].cpu().detach().numpy())
    # plt.show()

    imagemodel = ImageParameter(y.shape, zf).to(device)
    optimizer = torch.optim.Adam(imagemodel.parameters(), lr=lr)

    while True:
        if until_convergence:
            if torch.abs(prev_loss - reg_loss) < tol or iters > max_iters:
                break
        else:
            if iters > max_iters:
                break
        prev_loss = reg_loss
        optimizer.zero_grad()
        x = imagemodel().float().to(device)

        # plt.imshow(x[0,0].cpu().detach().numpy())
        # plt.colorbar()
        # plt.show()

        # Calculate Loss
        Ax = utils.hadamard_transform_torch(x, normalize=True) * mask
        dc_loss = (1-lmbda)*calc_dc(y, Ax)
        reg_loss = lmbda*(get_tv(x) + get_wavelets(x, device))
        # reg_loss = lmbda*(get_tv(x) + get_shearlets(x, shearlet))
        loss = dc_loss + reg_loss

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        iters += 1
        
    print('gd iters:', iters)
    return x.detach(), loss_list, y

def calc_dc(y, x):
    l2 = torch.nn.MSELoss(reduction='sum')
    return l2(y, x)

def get_tv(x):
    """Total variation loss

    Parameters
    ----------
    x : torch.Tensor (batch_size, img_height, img_width, 2)
        Input image

    Returns
    ----------
    tv_loss : TV loss
    """
    tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs())
    tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs())
    return tv_x + tv_y

def get_wavelets(x, device):
    """L1-penalty on wavelets

    Parameters
    ----------
    x : torch.Tensor (batch_size, img_height, img_width, 2)
        Input image

    Returns
    ----------
    tv_loss : wavelets loss
    """
    xfm = DWTForward(J=3, mode='zero', wave='db4').to(device) # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(x)

    batch_size = x.shape[0]
    channels = x.shape[1]
    rows = nextPowerOf2(Yh[0].shape[-2]*2)
    cols = nextPowerOf2(Yh[0].shape[-1]*2)
    wavelets = torch.zeros(batch_size, channels, rows, cols).to(device)
    # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
    for i, band in enumerate(Yh):
        irow = rows // 2**(i+1)
        icol = cols // 2**(i+1)
        wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

    wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
    return wavelets.norm(p=1)

def get_shearlets(x, shearlet):
    # x = x.squeeze()
    # x = torch.rfft(x, 2, normalized=True, onesided=False)
    scales = [0.5] * 2
    # shearlet = ShearletTransform(256, 256, scales)#, cache='/home/aw847/shear_cache/')
    shears = shearlet.forward(x)
    l1_shear = shears.norm(p=1)
    return l1_shear

def nextPowerOf2(n):
    """Get next power of 2"""
    count = 0;

    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;
def hqsplitting(xdata, mask, w_coeff, tv_coeff, lmbda, device, until_convergence, K, lr, max_iters=1000, tol=1e-8):
    y = xdata.clone()
    y_zf = utils.ifft(y)
    y, y_zf = utils.scale(y, y_zf)
    x = y_zf 
    
    final_losses = []
    final_dcs = []
    final_regs = []
    metrics = []
    for iteration in range(K):
        print('iteration:', iteration, lmbda, w_coeff, tv_coeff, lr)
        #  z-minimization
        z, loss_list = gradient_descent(x, until_convergence, max_iters, tol, w_coeff, tv_coeff, lmbda, lr, device, mask)
            
        # x-minimization
        z_ksp = utils.fft(z)
        x_ksp = losslayer.data_consistency(z_ksp, y, mask, lmbda=lmbda)
        x = utils.ifft(x_ksp)

        final_l, final_dc, final_reg = losslayer.final_loss(x, y, mask, w_coeff, tv_coeff, device)
        final_losses.append(final_l.item())
        final_dcs.append(final_dc.item())
        final_regs.append(final_reg.item())
    return x, final_losses, final_dcs, final_regs
