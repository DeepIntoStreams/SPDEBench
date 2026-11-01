# DLR for Phi42 with NORS-matched MFVs

This package uses exactly the same seven cached Phi42 base features, HDF5/MAT
data loader, split, target cache, optimizer, early stopping, metrics, and
evaluation timing code as `model.NORS.nors_phi42`.

DLR maps the seven base MFVs to a learned latent field, then rebuilds the four
trees that depend on the initial/latent field:

1. `latent`
2. `I[latent]`
3. `I[I[xi] * latent]`
4. `I[latent^2]`

The seven base MFVs are identical between NORS and DLR. The additional latent
channels are created inside DLR's forward pass and are included in model
inference time.

## Training

```bash
python -m model.DLR.dlr_phi42.train \
  --data-path "/workspace/results/Phi42+/datasetPhi42+_reno_xi_eps128_1200.mat" \
  --output-dir "/workspace/result/DLR_Phi42/seed_3407" \
  --checkpoint-name "DLR_Phi42_reno_xi_eps128_best_val.pt" \
  --max-time 250 \
  --sub-t 5 \
  --sub-x 1 \
  --diffusion 1.0 \
  --noise-scale 0.1 \
  --feature-batch-size 8 \
  --ntrain 840 \
  --nval 180 \
  --ntest 180 \
  --batch-size 10 \
  --epochs 1000 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --modes-x 8 \
  --modes-y 8 \
  --modes-t 8 \
  --width 8 \
  --depth 4 \
  --time-padding 6 \
  --scheduler-patience 50 \
  --scheduler-factor 0.1 \
  --scheduler-threshold 0.000001 \
  --early-stopping 80 \
  --delta 0.0002 \
  --print-every 20 \
  --seed 3407 \
  --device cuda
```

The same command accepts the streaming HDF5 datasets supported by NORS. Use
batch size one and gradient accumulation for the 128x128 dataset if required.

## Independent evaluation

```bash
python -m model.DLR.dlr_phi42.eval \
  --checkpoint "/workspace/result/DLR_Phi42/seed_3407/DLR_Phi42_reno_xi_eps128_best_val.pt" \
  --output-dir "/workspace/result/DLR_Phi42/seed_3407/evaluation" \
  --eval-data-paths \
    "/workspace/results/Phi42_evaluation_data/Phi42_test_seed10001_reno_xi_eps128_180.mat" \
    "/workspace/results/Phi42_evaluation_data/Phi42_test_seed10002_reno_xi_eps128_180.mat" \
    "/workspace/results/Phi42_evaluation_data/Phi42_test_seed10003_reno_xi_eps128_180.mat" \
    "/workspace/results/Phi42_evaluation_data/Phi42_test_seed10004_reno_xi_eps128_180.mat" \
    "/workspace/results/Phi42_evaluation_data/Phi42_test_seed10005_reno_xi_eps128_180.mat" \
  --modes1 8 \
  --modes2 8 \
  --modes3 8 \
  --width 8 \
  --skip-save-predictions \
  --delete-feature-cache-after-eval
```

Evaluation deliberately calls the NORS evaluator. It excludes the first batch,
synchronizes CUDA around every subsequent model forward, and reports seconds
per sample. `feature_avg_sample_time` measures the same cached seven-feature
construction as NORS; `inference_avg_sample_time` includes DLR's latent RS
stage; `end_to_end_avg_sample_time` is their sum.
