# DARNet: Deep Active Ray Network for Building Segmentation #

This repository contains code for the DARNet framework as described in our [CVPR 2019 paper](https://arxiv.org/abs/1905.05889).

## Dependencies
- Python 3 (we used v3.6.5 with Anaconda)
- PyTorch (we used v0.4.1)
- PIL
- scipy and associated packages
- tqdm

## Instructions
1. Download datasets from [here](https://github.com/dmarcosg/DSAC)
1. Unzip datasets to your desired directory
1. Modify `setup.ini` to reflect these directories, and the directory where you intend to keep results
1. Complete setup by running `./setup.sh`
1. Run experiments with `runner.py` (modify as necessary to run a subset, or to coordinate across different machines, etc.)

Please contact me at dominic@cs.toronto.edu for questions.