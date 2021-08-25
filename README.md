# Compressed Sensing Fluorescent Microscopy

## Data
Download FMD Dataset from https://github.com/yinhaoz/denoising-fluorescence and put the root folder in `denoising-fluorescent/`

## Train
### Usage
`python scripts/run.py -fp <experiment_name> --mask_type <mask_type>`

`<mask_type>` can be one of `[learned, random, equispaced, uniform, halfhalf]`.

### Example
`python scripts/run.py -fp example --mask_type learned`
