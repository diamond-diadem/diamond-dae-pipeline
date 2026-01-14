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

First, create and activate a virtual environment:

```bash
python3 -m venv .ddae-venv
source .ddae-venv/bin/activate  # On Windows use: .ddae-venv\Scripts\activate
```

Then install the package and its dependencies using `pip3`:

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

## Model Overview (src)

The denoising model lives in `src/ddae1d/model.py` as `DiamondDAE1D`, a 1D convolutional denoising autoencoder designed for Noise2Noise training. It is built as:

- **Encoder/decoder blocks:** `n_conv_blocks` total, split evenly between encoder and decoder. The encoder uses `Conv1d` layers followed by `MaxPool1d`; the decoder mirrors this with `ConvTranspose1d` and per-block strides.
- **Filters list:** `filters` defines the encoder channel sizes. Its length must be `n_conv_blocks // 2`.
- **Latent block:** if `n_conv_blocks` is odd, a latent stack is inserted with `latent_dim` and `n_latent_layers`.
- **Activations:** `activations` is resolved from `torch.nn.functional` (e.g., `"relu"`). The output activation is selected by `output_activation` (`"linear"`, `"sigmoid"`, `"tanh"`, `"relu"`).
- **Input shape:** training expects tensors shaped `(batch, 1, length)`. The `predict` helper accepts `(n_samples, length)` or `(n_samples, n_realizations, length)` and reshapes internally.

Noise2Noise training uses `Noise2NoiseDataset`, which dynamically samples two different noisy realizations per spectrum for input/target pairing.

## How to Configure the Model

Model and training settings are configured in `notebooks/training/config.json`:

- `model_params`: architectural choices (`n_conv_blocks`, `filters`, `kernel_size`, `latent_dim`, `n_conv_per_block`, `n_latent_layers`, `use_batchnorm`, `dropout_rate`, `activations`, `output_activation`).
- `training_params`: runtime settings (`batch_size`, `epochs`, `learning_rate`, `device`, `verbose`).
- `model_filename`: output filename saved under `models/`.

When you change `model_filename`, update `notebooks/denoising/config.json` so the denoising notebook loads the same trained model. If you adjust `filters`, ensure it stays aligned with `n_conv_blocks // 2`, and keep `kernel_size` odd to match the model's validation checks.

## Configuration Notes

- Update preprocessing parameters in `notebooks/preprocessing/*/config.json`.
- Adjust training hyperparameters and output model filename in `notebooks/training/config.json`.
- Set the model filename used for denoising in `notebooks/denoising/config.json`.

## How to Cite

If you use this software, please cite the associated paper:

"A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy", Advanced Engineering Materials (2026).

For full citation metadata, see `CITATION.cff`.
