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
