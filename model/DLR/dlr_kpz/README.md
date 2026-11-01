# DLR for KPZ with NORS-matched MFVs

This package uses exactly the same seven cached KPZ base features, data loader,
split, optimizer, early stopping, metrics, and evaluation timing code as
`model.NORS.nors_kpz`.

DLR first maps the seven base MFVs to a learned latent field. It then rebuilds
the four trees that depend on the initial/latent field:

1. `latent`
2. `I[latent]`
3. `I[(D I[xi])(D latent)]`
4. `I[(D latent)^2]`

The seven base MFVs are therefore identical between NORS and DLR; the four
additional channels are created inside DLR's forward pass and are included in
model inference time.

## Training

Use the same arguments as NORS, changing only the module and output names:

```bash
python -m model.DLR.dlr_kpz.train \
  --data-path "/workspace/results/KPZ+/KPZ+renosigma01_xi_lam005_trc128_1200.mat" \
  --output-dir "/workspace/result/DLR_KPZ/seed_3407" \
  --checkpoint-name "DLR_KPZ_sigma01_lam005_trc128_best_val.pt" \
  --max-time 51 \
  --sub-t 1 \
  --sub-x 1 \
  --spatial-points 128 \
  --diffusion 1.0 \
  --noise-scale 0.1 \
  --renorm-constant 128 \
  --feature-batch-size 64 \
  --ntrain 840 \
  --nval 180 \
  --ntest 180 \
  --batch-size 10 \
  --epochs 1000 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --modes-x 16 \
  --modes-t 16 \
  --width 8 \
  --depth 4 \
  --time-padding 6 \
  --scheduler-patience 50 \
  --scheduler-factor 0.1 \
  --early-stopping 80 \
  --delta 0.0002 \
  --print-every 20 \
  --seed 3407 \
  --device cuda
```

## Evaluation

```bash
python -m model.DLR.dlr_kpz.eval \
  --checkpoint "/workspace/result/DLR_KPZ/seed_3407/DLR_KPZ_sigma01_lam005_trc128_best_val.pt" \
  --output-dir "/workspace/result/DLR_KPZ/seed_3407/evaluation" \
  --eval-data-paths \
    "/path/to/KPZ_test_seed10001_180.mat" \
    "/path/to/KPZ_test_seed10002_180.mat" \
    "/path/to/KPZ_test_seed10003_180.mat" \
    "/path/to/KPZ_test_seed10004_180.mat" \
    "/path/to/KPZ_test_seed10005_180.mat" \
  --noise-scale 0.1 \
  --renorm-constant 128 \
  --modes-x 16 \
  --modes-t 16 \
  --width 8 \
  --skip-save-predictions \
  --delete-feature-cache-after-eval
```

Evaluation deliberately calls the NORS evaluator. It excludes the first batch,
synchronizes CUDA around every subsequent model forward, and reports seconds
per sample. `feature_avg_sample_time` measures the same cached seven-feature
construction as NORS; `inference_avg_sample_time` includes DLR's latent RS
stage; `end_to_end_avg_sample_time` is their sum.
