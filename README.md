# DIAMOND Denoising Pipeline for 1D spectra 

This repository contains the complete, reproducible implementation of the denoising pipeline presented in the article “A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy”.

## Requirements

To run the denoising pipeline, install the main packages listed below. The `requirements.txt` file includes these packages and their dependencies:

```
numpy==2.4.0
pandas==2.3.3
matplotlib==3.10.8
ipykernel==7.1.0
scipy==1.16.3
scikit-learn==1.8.0
torch==2.9.1
h5py==3.15.1
optuna==4.6.0
```

## Installation

Install the package and its dependencies using `pip3`:

```bash
pip3 install -r requirements.txt
pip3 install -e .
```