# Compressed Sensing Fluorescence Microscopy (MICCAI 2021)

Code for MICCAI 2021 paper "Joint Optimization of Hadamard Sensing and Reconstruction in Compressed Sensing Fluorescence Microscopy", [arXiv.2105.07961](https://arxiv.org/abs/2105.07961).

This repository runs code for the above paper on the publicly-available [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence).

## Data
Download FMD Dataset from https://github.com/yinhaoz/denoising-fluorescence and put the root folder in `denoising-fluorescent/`

## Train
### Usage
`python scripts/run.py -fp <experiment_name> --mask_type <mask_type>`

`<mask_type>` can be one of `[learned, random, equispaced, uniform, halfhalf]`.

### Example
`python scripts/run.py -fp example --mask_type learned`

Model checkpoints and other arguments are saved to `out/`.
