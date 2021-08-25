# Joint Optimization of Hadamard Sensing and Reconstruction in Compressed Sensing Fluorescence Microscopy (MICCAI 2021)
Alan Q. Wang, Aaron K. LaViolette, Leo Moon, Chris Xu, and Mert R. Sabuncu.

[Link to paper](https://arxiv.org/abs/2105.07961)

## Data
Download FMD Dataset from https://github.com/yinhaoz/denoising-fluorescence and put the root folder in `denoising-fluorescent/`

## Train
### Usage
`python scripts/run.py -fp <experiment_name> --mask_type <mask_type>`

`<mask_type>` can be one of `[learned, random, equispaced, uniform, halfhalf]`.

### Example
`python scripts/run.py -fp example --mask_type learned`
