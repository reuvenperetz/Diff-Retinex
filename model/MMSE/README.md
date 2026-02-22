# MMSE pre-processing for Retinex components

This module provides simple MMSE pre-processing models (small UNet or CNN) that can be inserted
between the TDN decomposition and the diffusion models (Diff-RDA/Diff-IDA).

## Dataset format

```
mmse_dataset/
  train/
    low/
    R/   # optional reflectance targets
    L/   # optional illumination targets
    high/
    R_high/
    L_high/
  val/
    low/
    R/
    L/
    high/
    R_high/
    L_high/
```

## Train

Reflectance (R -> R_high, 3ch -> 3ch):
```
python train_mmse.py --data-root /path/to/mmse_dataset --train-r --arch unet
```

Illumination (L -> L_high, 1ch -> 1ch):
```
python train_mmse.py --data-root /path/to/mmse_dataset --train-l --arch unet
```

## Loss
MMSE optimizes MSE on the component, optionally plus reconstruction MSE:
```
loss = MSE(component_pred, component_high) + recon_weight * MSE(recon, high)
```
Set `--recon-weight` to enable the reconstruction term.

## Validation progress images
Save input/pred/gt for selected validation indices every N epochs:
```
python train_mmse.py --data-root /path/to/mmse_dataset --train-l \
  --val-vis-every 5 --val-vis-indices 0,1 --val-vis-dir mmse_vis
```

## Validation metrics
During training, validation runs compute PSNR/SSIM and (if available) LPIPS and NIQE.
Install `lpips` and a `scikit-image` version that exposes `skimage.metrics.niqe` if you want those metrics.

## MLflow logging
Enable MLflow to log parameters, metrics, and artifacts (checkpoints + validation visuals):
```
python train_mmse.py --data-root /path/to/mmse_dataset --train-l --mlflow \
  --mlflow-exp MMSE --mlflow-run my_run
```

Both:
```
python train_mmse.py --data-root /path/to/mmse_dataset --train-r --train-l --arch unet
```

## Build MMSE dataset from TDN
This generates `R/` and `L/` targets from the existing TDN weights and copies `low/` images.
Use `--include-high` to also decompose `high/` into `R_high/` and `L_high/`.

```
python build_mmse_dataset.py \
  --input-root /path/to/dataset \
  --output-root /path/to/mmse_dataset \
  --tdn-weights model/Diff_TDN/weights/checkpoint_LOL_Diff_TDN.pth \
  --use-eval-as-val \
  --include-high
```

## Inference integration
Use the flags in `test_from_dataset.py`:
```
python test_from_dataset.py \
  --use_mmse --mmse_components both \
  --mmse_arch unet \
  --mmse_r_weights model/MMSE/weights/mmse_r.pth \
  --mmse_l_weights model/MMSE/weights/mmse_l.pth
```
