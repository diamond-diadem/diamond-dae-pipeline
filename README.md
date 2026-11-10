# DIAMOND Denoising Pipeline for 1D spectra

This repository contains the complete, reproducible implementation of the denoising pipeline presented in the article “A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy”.

## How to Cite

If you use this software, please cite the associated paper:

"A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy", Advanced Engineering Materials (2026).

For full citation metadata, see `CITATION.cff`.

## Requirements

This pipeline requires Python >=3.13. Direct dependencies are declared in `pyproject.toml`; exact versions of the full dependency tree are pinned in `uv.lock` for reproducible installs.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and reproducible installs.

From the repository root, create the environment and install the exact locked dependency versions:

```bash
uv sync
```

## Developer Notes

Every `.ipynb` file in the repository goes through a git clean filter ([scripts/strip_notebooks.py](scripts/strip_notebooks.py)) that keeps only cell sources, attachments and tags: outputs, execution counts, cell ids, cell and notebook metadata (kernel, interpreter, widgets, editor-specific keys) are dropped or normalized. Running a notebook in Jupyter, JupyterLab, VS Code or any other editor therefore never shows up as a change. Enable it once per clone:

```bash
git config filter.stripnotebook.clean "python3 scripts/strip_notebooks.py"
git config filter.stripnotebook.smudge cat
git config filter.stripnotebook.required true
```

## Preparing the Data

Before running the preprocessing notebooks, ensure that the raw data is saved or converted into a `.npy` file. The training dataset should have the shape `(number of spatial points, number of repetitions per spatial point, number of spectral points)`. The final map to be denoised should have the shape `(number of points along x, number of points along y, number of spectral points)`.

Data used in the associated study is available at [doi.org/10.5281/zenodo.18244161](https://doi.org/10.5281/zenodo.18244161). Place the downloaded files in `data/from-zenodo/` before running the preparation steps for the pipeline. (notebooks in `prepare-data-from-zenodo/`).

## Repository Layout

- `data/raw/trainset/` and `data/raw/final-map/`: raw `.npy` inputs referenced by the preprocessing configs.
- `data/preprocessed/`: intermediate outputs from preprocessing.
- `models/`: saved denoising models.
- `data/results/denoised/`: denoised final map outputs.
- `notebooks-pipeline/`: end-to-end pipeline notebooks (preprocessing, training, denoising, clustering).

## Pipeline Workflow

Run the notebooks in the order below. Each notebook reads its configuration from the adjacent `config.json`.

1. `notebooks-pipeline/preprocessing/trainset/trainset-preprocessing.ipynb`
   - Input: `data/raw/trainset/raw-trainset.npy`
   - Output: `data/preprocessed/trainset/noisy.npy`
2. `notebooks-pipeline/preprocessing/final-map/final-map-preprocessing.ipynb`
   - Input: `data/raw/final-map/raw-final-map.npy`
   - Output: `data/preprocessed/final-map/preprocessed-final-map.npy`
3. `notebooks-pipeline/training/training.ipynb`
   - Input: `data/preprocessed/trainset/noisy.npy`
   - Output: `models/diamond-dae1d-noise2noise.model`
4. `notebooks-pipeline/denoising/denoising-final-map.ipynb`
   - Inputs: `data/preprocessed/final-map/preprocessed-final-map.npy`, `models/diamond-dae1d-noise2noise.model`
   - Output: `data/results/denoised/denoised-final-map.npy`
5. `notebooks-pipeline/clustering/clustering.ipynb` (optional)
   - Use to explore or segment denoised spectra after the main pipeline.

## Model Overview (src)

The denoising model lives in `src/ddae1d/model.py` as `DiamondDAE1D`, a 1D convolutional denoising autoencoder designed for Noise2Noise training. It is built as:

- **Encoder/decoder blocks:** `n_conv_blocks` total, split evenly between encoder and decoder. The encoder uses `Conv1d` layers followed by `MaxPool1d`; the decoder mirrors this with `ConvTranspose1d` and per-block strides.
- **Filters list:** `filters` defines the encoder channel sizes. Its length must be `n_conv_blocks // 2`.
- **Latent block:** if `n_conv_blocks` is odd, a latent stack is inserted with `latent_dim` and `n_latent_layers`.
- **Activations:** `activations` (hidden layers) accepts any callable, or any name resolved dynamically from `torch.nn.functional` then `torch` (e.g., `"relu"`). `output_activation` is resolved differently and more narrowly: it is limited to a fixed, hardcoded set of 4 names (`"linear"`, `"sigmoid"`, `"tanh"`, `"relu"`) and does not accept a callable or any other `torch`/`torch.nn.functional` name.
- **Input shape:** training expects tensors shaped `(batch, 1, length)`. The `predict` helper accepts `(n_samples, length)` or `(n_samples, n_realizations, length)` and reshapes internally.

Noise2Noise training uses `Noise2NoiseDataset`, which dynamically samples two different noisy realizations per spectrum for input/target pairing.

## How to Configure the Model

Model and training settings are configured in `notebooks-pipeline/training/config.json`:

- `model_params`: architectural choices (`n_conv_blocks`, `filters`, `kernel_size`, `latent_dim`, `n_conv_per_block`, `n_latent_layers`, `use_batchnorm`, `dropout_rate`, `activations`, `output_activation`).
- `training_params`: runtime settings (`batch_size`, `epochs`, `learning_rate`, `device`, `verbose`).
- `model_filename`: output filename saved under `models/`.

### Model Parameters

- `n_conv_blocks`: total encoder+decoder blocks; encoder and decoder each use `n_conv_blocks // 2`.
- `filters`: list of encoder channels per block (length must equal `n_conv_blocks // 2`). The decoder mirrors this list in reverse.
- `kernel_size`: convolution kernel width; must be odd and >= 1 to preserve alignment with padding.
- `latent_dim`: channel size for the latent stack; defaults to the last `filters` value if omitted.
- `n_conv_per_block`: number of `Conv1d` layers per encoder block before pooling (mirrored in the decoder).
- `n_latent_layers`: number of `Conv1d` layers in the latent stack (only used when `n_conv_blocks` is odd).
- `use_batchnorm`: toggles `BatchNorm1d` after each convolution.
- `dropout_rate`: dropout probability applied after convolutions when > 0.
- `activations`: hidden-layer activation; any callable, or any name resolved from `torch.nn.functional` then `torch` (e.g., `"relu"`).
- `output_activation`: output nonlinearity; restricted to exactly `"linear"`, `"sigmoid"`, `"tanh"`, or `"relu"` — unlike `activations`, this is not resolved dynamically and any other value raises a `ValueError`.
- `strides` (optional): per-encoder-block pooling stride(s). The model accepts an int or a list with length `n_conv_blocks // 2`, defaulting to `2` if not provided.

### Training Parameters

- `batch_size`: training batch size.
- `epochs`: number of training epochs.
- `learning_rate`: optimizer learning rate used by the training routine.
- `device`: `"cuda"` or `"cpu"` (or a specific device string) to control where tensors are placed.
- `verbose`: verbosity level for training logs.

When you change `model_filename`, update `notebooks-pipeline/denoising/config.json` so the denoising notebook loads the same trained model. If you adjust `filters`, ensure it stays aligned with `n_conv_blocks // 2`, and keep `kernel_size` odd to match the model's validation checks. If you add `strides`, ensure its length matches the encoder depth.

## Configuration Notes

- Update preprocessing parameters in `notebooks-pipeline/preprocessing/*/config.json`.
- Adjust training hyperparameters and output model filename in `notebooks-pipeline/training/config.json`.
- Set the model filename used for denoising in `notebooks-pipeline/denoising/config.json`.
- The offset profile (`apply_offset`) is a low per-channel percentile across all spectra, so it assumes at least `offset_percentile` % of the spectra are free of Raman signal on every channel. A band present almost everywhere would end up in the offset and be subtracted from every spectrum: check the offset profile plot for peaks.
