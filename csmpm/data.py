"""
Utility functions for CSFM.
For more details, please read:
    Alan Q. Wang, Aaron K. LaViolette, Leo Moon, Chris Xu, and Mert R. Sabuncu.
    "Joint Optimization of Hadamard Sensing and Reconstruction in Compressed Sensing Fluorescence Microscopy." 
    MICCAI 2021

See also: https://github.com/yinhaoz/denoising-fluorescence/blob/master/denoising/utils/data_loader.py
"""
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import _is_pil_image
from torchvision.datasets.folder import has_file_allowed_extension
import json

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
                    noisy_captures = []  # Contains all captures for single FOV
                    for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                        if is_image_file(fname):
                            noisy_file = os.path.join(noisy_fov_dir, fname)
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
                noisy.append(self.transform(n))

        noisy = torch.stack(noisy, dim=0)
        return noisy, clean

    def __len__(self):
        return len(self.samples)


def load_denoising(root, train, batch_size, types=None, captures=2,
                   patch_size=256, transform=None, target_transform=None, loader=pil_loader,
                   test_fov=19):
    """
    Args:
        root (str): root directory to dataset
        train (bool): train or test
        batch_size (int): e.g. 4
        types (seq, None): e.g. [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.FiveCrop(patch_size),
            transforms.Lambda(lambda crops: torch.stack([
                fluore_to_tensor(crop) for crop in crops])),
        ])
    target_transform = transform

    dataset = FolderNoisy(root, train,
                          types=types, test_fov=test_fov,
                          captures=captures, transform=transform,
                          target_transform=target_transform, loader=pil_loader)
    kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=False, **kwargs)

    return data_loader
