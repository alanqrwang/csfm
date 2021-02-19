import os
import numpy as np
from PIL import Image
import numbers
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor, _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension
import sys
import json
from pprint import pprint
from time import time

__all__ = ['fluore_to_tensor', 'DenoisingFolder', 'DenoisingFolderN2N', 
           'DenoisingTestMixFolder', 'load_denoising', 
           'load_denoising_n2n_train', 'load_denoising_test_mix']

IMG_EXTENSIONS = ('.png')

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path):
    img = Image.open(path)
    return img


def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.
    
    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not(_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))
    img = torch.from_numpy(np.array(pic))
    img = img.squeeze(-1).unsqueeze(0)

    # # handle PIL Image
    # if pic.mode == 'I':
    #     img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    # elif pic.mode == 'I;16':
    #     img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    # elif pic.mode == 'F':
    #     img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    # elif pic.mode == '1':
    #     img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    # else:
    #     # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
    #     img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    # if pic.mode == 'YCbCr':
    #     nchannel = 3
    # elif pic.mode == 'I;16':
    #     nchannel = 1
    # else:
    #     nchannel = len(pic.mode)

    # img = img.view(pic.size[1], pic.size[0], nchannel)
    
    # if nchannel == 1:
    #     img = img.squeeze(-1).unsqueeze(0)
    # elif pic.mode in ('RGB', 'RGBA'):
    #     # RBG to grayscale: 
    #     # https://en.wikipedia.org/wiki/Luma_%28video%29
    #     ori_dtype = img.dtype
    #     rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
    #     img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
    #     img = img.to(ori_dtype)
    # else:
    #     # other type not supported yet: YCbCr, CMYK
    #     raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img

class FolderNoisy(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples
    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root, train, types=None, test_fov=19,
        captures=50, transform=None, target_transform=None, loader=pil_loader):
        super().__init__()
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        self.noise_levels = [1]
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        self.train = train
        if train:
            fovs = list(range(1, 20+1))
            fovs.remove(test_fov)
            self.fovs = fovs
        else:
            self.fovs = [test_fov]
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train' if train else 'test',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        # types: microscopy_cell
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    noisy_captures = [] # Contains all captures for single FOV
                    for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                        if is_image_file(fname):
                            noisy_file = os.path.join(noisy_fov_dir, fname)
                            noisy_captures.append(noisy_file)
                            # samples.append((noisy_file, clean_file))
                    # randomly select one noisy capture when loading from FOV     
                    samples.append((noisy_captures, clean_file))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        clean = self.loader(clean_file)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        noisy = []
        for f in noisy_file:
            n = self.loader(f)
            if self.transform is not None:
                noisy.append(self.transform(n))

        noisy = torch.stack(noisy, dim=0)
        return noisy, clean

    def __len__(self):
        return len(self.samples)


class TestMixFolderNoisy(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, transform, target_transform):
        super().__init__()
        self.noise_levels = [1]
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_mix',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))


    
    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        
        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            noisy_captures = []
            for fname in sorted(os.listdir(noise_dir)):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    noisy_captures.append(noisy_file)
            samples.append((noisy_captures, clean_file))

        return samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        clean = self.loader(clean_file)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        noisy = []
        for f in noisy_file:
            n = self.loader(f)
            if self.transform is not None:
                transformed = self.transform(n)
                noisy.append(self.transform(n))

        noisy = torch.stack(noisy, dim=0)
        return noisy, clean

    def __len__(self):
        return len(self.samples)

class Folder(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with 
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples
    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in 
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes 
            in the target and transforms it.
        loader (callable, optional): image loader
    """
    def __init__(self, root, train, types=None, test_fov=19,
        captures=50, transform=None, target_transform=None, loader=pil_loader):
        super().__init__()
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        self.img_params = {'Confocal_BPAE_B': (1.39e-2, 0),\
                  'Confocal_BPAE_G': (1.37e-2, 0),\
                  'Confocal_BPAE_R': (1.21e-2, 0),\
                  'Confocal_MICE': (1.94e-2, 0),\
                  'Confocal_FISH': (9.43e-2, 0),\
                  'TwoPhoton_BPAE_B': (3.31e-2, 0),\
                  'TwoPhoton_BPAE_G': (2.55e-2, 0),\
                  'TwoPhoton_BPAE_R': (2.1e-2, 0),\
                  'TwoPhoton_MICE': (3.38e-2, 0),\
                  'WideField_BPAE_B': (2.29e-4, 2.35e-4),\
                  'WideField_BPAE_G': (1.94e-3, 1.91e-4),\
                  'WideField_BPAE_R': (3.55e-4, 1.95e-4)\
                  }
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        self.train = train
        if train:
            fovs = list(range(1, 20+1))
            fovs.remove(test_fov)
            self.fovs = fovs
        else:
            self.fovs = [test_fov]
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'train' if train else 'test',
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))


    def _get_imaging_params(self, img_type):
        assert img_type in self.img_params, 'invalid img_type'
        return self.img_params[img_type]

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        # types: microscopy_cell
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
            if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            a, b = self._get_imaging_params(subdir.split('/')[-1])
            for i_fov in self.fovs:
                clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                samples.append((clean_file, a, b))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clean)
        """
        clean_file, a, b = self.samples[index]
            
        clean = self.loader(clean_file)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        a = torch.as_tensor(a).repeat(len(clean)).float()
        b = torch.as_tensor(b).repeat(len(clean)).float()

        return clean, a, b

    def __len__(self):
        return len(self.samples)

class TestMixFolder(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, transform, target_transform):
        super().__init__()
    
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.img_params = {'Confocal_BPAE_B': (1.39e-2, 0),\
                  'Confocal_BPAE_G': (1.37e-2, 0),\
                  'Confocal_BPAE_R': (1.21e-2, 0),\
                  'Confocal_MICE': (1.94e-2, 0),\
                  'Confocal_FISH': (9.43e-2, 0),\
                  'TwoPhoton_BPAE_B': (3.31e-2, 0),\
                  'TwoPhoton_BPAE_G': (2.55e-2, 0),\
                  'TwoPhoton_BPAE_R': (2.1e-2, 0),\
                  'TwoPhoton_MICE': (3.38e-2, 0),\
                  'WideField_BPAE_B': (2.29e-4, 2.35e-4),\
                  'WideField_BPAE_G': (1.94e-3, 1.91e-4),\
                  'WideField_BPAE_R': (3.55e-4, 1.95e-4)\
                  }
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_mix',
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))


    def _get_imaging_params(self, img_type):
        img_type = img_type[:-6]
        assert img_type in self.img_params, 'invalid img_type'
        return self.img_params[img_type]
    
    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')
        
        for fname in sorted(os.listdir(gt_dir)):
            a, b = self._get_imaging_params(fname)
            if is_image_file(fname):
                clean_file = os.path.join(gt_dir, fname)
                samples.append((clean_file, a, b))

        return samples


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        clean_file, a, b = self.samples[index]
        clean = self.loader(clean_file)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        a = torch.as_tensor(a).repeat(len(clean)).float()
        b = torch.as_tensor(b).repeat(len(clean)).float()

        return clean, a, b

    def __len__(self):
        return len(self.samples)


def load_denoising(root, train, batch_size, get_noisy, types=None, captures=2,
    patch_size=256, transform=None, target_transform=None, loader=pil_loader,
    test_fov=19):
    """
    files: root/type/noise_level/fov/captures.png
        total 12 x 5 x 20 x 50 = 60,000 images
        raw: 12 x 20 x 50 = 12,000 images
    
    Args:
        root (str): root directory to dataset
        train (bool): train or test
        batch_size (int): e.g. 4
        get_noisy (bool): whether to load noisy frames or just ground truth
        types (seq, None): e.g. [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        # default to center crop the image from 512x512 to 256x256
        transform = transforms.Compose([
            transforms.FiveCrop(patch_size),
            transforms.Lambda(lambda crops: torch.stack([
                fluore_to_tensor(crop) for crop in crops])),
            # fluore_to_tensor,
            # transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])
    target_transform = transform
        
    if get_noisy:
        dataset = FolderNoisy(root, train, 
            types=types, test_fov=test_fov,
            captures=captures, transform=transform, 
            target_transform=target_transform, loader=pil_loader)
    else:
        dataset = Folder(root, train, 
            types=types, test_fov=test_fov,
            captures=captures, transform=transform, 
            target_transform=target_transform, loader=pil_loader)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader
    

def load_denoising_test_mix(root, batch_size, get_noisy, loader=pil_loader, 
    transform=None, target_transform=None, patch_size=256):
    """
    files: root/test_mix/noise_level/captures.png
        
    Args:
        root (str):
        batch_size (int): 
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.FiveCrop(patch_size),
            transforms.Lambda(lambda crops: torch.stack([
                fluore_to_tensor(crop) for crop in crops])),
            # fluore_to_tensor,
            # transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])
    # the same
    target_transform = transform
        
    if get_noisy:
        dataset = TestMixFolderNoisy(root, loader, transform, 
            target_transform)
    else:
        dataset = TestMixFolder(root, loader, transform, 
            target_transform)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        shuffle=True, drop_last=False, **kwargs)

    return data_loader
