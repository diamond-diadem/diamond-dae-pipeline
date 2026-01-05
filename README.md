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

## Developer Notes

To keep notebooks clean for public commits, enable the repo hook once:

```bash
git config core.hooksPath .githooks
```

## Preparing the Data

Before running the preprocessing notebooks, ensure that the raw data is saved or converted into a `.npy` file. The training dataset should have the shape `(number of spatial points, number of repetitions per spatial point, number of spectral points)`. The final map to be denoised should have the shape `(number of spatial points, number of spectral points)`.

## Repository Layout

- `data/raw/trainset/` and `data/raw/final-map/`: raw `.npy` inputs referenced by the preprocessing configs.
- `data/preprocessed/`: intermediate outputs from preprocessing.
- `models/`: saved denoising models.
- `data/results/denoised/`: denoised final map outputs.
- `notebooks/`: end-to-end pipeline notebooks (preprocessing, training, denoising, clustering).

## Pipeline Workflow

Run the notebooks in the order below. Each notebook reads its configuration from the adjacent `config.json`.

1. `notebooks/preprocessing/trainset/trainset-preprocessing.ipynb`
   - Input: `data/raw/trainset/raw-trainset.npy`
   - Output: `data/preprocessed/trainset/noisy.npy`
2. `notebooks/preprocessing/final-map/final-map-preprocessing.ipynb`
   - Input: `data/raw/final-map/raw-final-map.npy`
   - Output: `data/preprocessed/final-map/preprocessed-final-map.npy`
3. `notebooks/training/training.ipynb`
   - Input: `data/preprocessed/trainset/noisy.npy`
   - Output: `models/diamond-dae1d-noise2noise.model`
4. `notebooks/denoising/denoising-final-map.ipynb`
   - Inputs: `data/preprocessed/final-map/preprocessed-final-map.npy`, `models/diamond-dae1d-noise2noise.model`
   - Output: `data/results/denoised/denoised-final-map.npy`
5. `notebooks/clustering/clustering.ipynb` (optional)
   - Use to explore or segment denoised spectra after the main pipeline.

## Configuration Notes

- Update preprocessing parameters in `notebooks/preprocessing/*/config.json`.
- Adjust training hyperparameters and output model filename in `notebooks/training/config.json`.
- Set the model filename used for denoising in `notebooks/denoising/config.json`.
