# SwinIR-based High-Resolution Temperature Downscaling over the Korean Peninsula

This repository contains the code accompanying the manuscript:

> Ko, J., Jun, S.-Y., & Kug, J.-S. SwinIR-based high-resolution downscaling of temperature over the Korean Peninsula. *Geosciences* (In Revisiom).

A SwinIR-based deep-learning super-resolution (SR) framework that downscales low-resolution ERA5 reanalysis fields (0.25°) to the 1.5-km daily-mean 1.5-m temperature field of the Korea Meteorological Administration's Local Data Assimilation and Prediction System (LDAPS). The model fuses static topographic information through a dedicated Topography Encoder using a dual-fusion (LR-stage + HR-stage) structure.

## Repository Structure

```
SwinIR_downscaling/
├── Data/
│   ├── Data_processing.ipynb        # ERA5 / topography / LAI preprocessing & visualization
│   ├── topography_translate.sh      # GDAL + CDO workflow for DEM/slope remapping
│
├── Train/
│   ├── config.yaml                  # Central configuration (paths, hyperparameters, model)
│   ├── model_SwinIR_v3.py           # Model: SwinIR_v3 and SwinIR_v3_wo_topo
│   ├── Dataset.py                   # ERA5ToLDAPSDataset
│   ├── Dataloader.py                # Dataloaders, k-fold time-index splitting
│   ├── loss.py                      # SSIM loss, gradient loss (+ GAN losses)
│   ├── utils.py                     # Resolution-dependent path & normalization helpers
│   ├── train_SwinIR_v3.py           # Training entry point
│   ├── train_each_kfold.sh          # Run training across k-folds
│   ├── inference.py                 # Inference on the LDAPS test period
│   ├── run_inference.sh             # Sequential inference across k-folds
│   ├── inference_multi.sh           # Parallel multi-GPU inference
│   └── paleo_inference.py           # Application to CESM paleoclimate output
│
└── Analysis/
    └── Analysis_visualization.ipynb # Metric computation & figure generation
```

> **Note:** Input `.pt`/`.nc` datasets and model checkpoints are **not** included in this repository (see *Data* below). Expected data paths are defined in `Train/config.yaml` under `data_paths`.

## Requirements

- Python 3.x, PyTorch
- einops, PyYAML, numpy, tqdm, torchvision, torchinfo
- wandb (training logging)
- xarray, rioxarray, polars (preprocessing & analysis)
- GDAL, CDO (topography preprocessing)
- NVIDIA GPU (experiments were run on an NVIDIA H100 80GB GPU)

## Data

This repository does not redistribute any source data. The included station CSV files under `Data/` are derived station records; the gridded ERA5, DEM, SRTM, and LDAPS data must be obtained from their original providers and preprocessed into the `.pt` tensors expected by the dataloader.

| Dataset | Role | Source |
|---|---|---|
| ERA5 (T2m, U10, V10, SP, LAI) | LR dynamic / static input | [Copernicus CDS](https://cds.climate.copernicus.eu/) |
| Copernicus GLO-90 DEM (elevation) | HR static input | [Copernicus Data Space](https://dataspace.copernicus.eu/) |
| SRTM (slope) | HR static input | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) |
| LDAPS (1.5-m temperature) | HR target | [KMA Data Portal](https://data.kma.go.kr/) |
| ASOS / AWS | Independent validation | [KMA Data Portal](https://data.kma.go.kr/) |

Preprocessing steps are provided in `Data/Data_processing.ipynb` (gridded variables) and `Data/topography_translate.sh` (DEM/slope clipping, GeoTIFF→NetCDF conversion, conservative remapping to the 1.5-km target grid).

## Configuration

All paths and hyperparameters are controlled through `Train/config.yaml`. Key settings:

| Key | Default | Description |
|---|---|---|
| `input_vars` | `[t2m, u10, v10, sp]` | LR dynamic input variables |
| `static_vars` | `[elevation, slope, mask, latlon_LR, latlon_HR, lai]` | Static inputs |
| `model` | `swinir` | Model type |
| `data_processing_clim` | `residual_f` | Target preprocessing (spatial-perturbation field) |
| `criterion` | `mae+grad+ssim` | Loss combination |
| `ssim_lambda` / `grad_lambda` | `0.1` / `0.08` | Loss weights |
| `upsteps` | `4` | Number of upsampling blocks (DUSB), adjusted by input resolution |
| `upsample_mode` | `dual` | Dual upsampling (bilinear + pixel shuffle) |
| `epochs` / `batch_size` / `lr` | `100` / `16` / `2e-4` | Training schedule |
| `optimizer` / `scheduler` | `adam` / `CosineAnnealingLR` | Optimization |

Command-line arguments (`--res`, `--gpu`, `--kfold`, `--model`, `--version`) override the corresponding config values.

## Usage

All commands below are run from the `Train/` directory.

### 1. Training

Train a single fold:

```bash
python train_SwinIR_v3.py --config ./config.yaml --model swinir --res 1 --kfold 0 --version v1
```

Train across a range of folds (6-fold cross-validation, 2013–2024):

```bash
# bash train_each_kfold.sh <version> <start_fold> <end_fold>
bash train_each_kfold.sh v1 0 5
```

### 2. Inference (LDAPS test period)

```bash
# bash run_inference.sh <model> <res> <start_fold> <end_fold>
bash run_inference.sh swinir 1 0 5
```

For parallel multi-GPU inference across folds:

```bash
bash inference_multi.sh
```

### 3. Paleoclimate application (CESM past1myr)

```bash
python paleo_inference.py --config ./config.yaml --ckpt <path_to_checkpoint> --res 2
```

### 4. Analysis & figures

Open `Analysis/Analysis_visualization.ipynb` to reconstruct full temperature fields from the predicted spatial-perturbation fields, compute evaluation metrics (ACC, RMSE, MBE, STD ratio, SSIM, PC), and generate the manuscript figures.

## Model Overview

`model_SwinIR_v3.py` defines two variants:

- **`SwinIR_v3`** — the full model with the Topography Encoder.
- **`SwinIR_v3_wo_topo`** — the ablation variant for the topography-removal experiment, used to isolate the contribution of topographic information.

Architecture components: sinusoidal positional embedding for coordinates, a shallow feature extraction block, four Residual Swin Transformer Blocks (RSTB) with window / shifted-window attention, a Topography Encoder fused at both LR and HR stages, Dual Upsampling Blocks (bilinear + pixel-shuffle branches), and a final fusion block producing the temperature spatial-perturbation field.


## Contact

For questions about the code, please open an issue or contact the corresponding author, Jong-Seong Kug (jskug@snu.ac.kr).
