# Usage

## Data

The loader expects RGB images and matching `.mat` labels with an `inst_map` array:

```text
data/<monuseg or tnbc>/
  train_12/images/
  train_12/labels/
  test/images/
  test/labels/
```

Place initialization and trained checkpoints in `checkpoints/`.

## Training

```bash
python main.py --data_path data/monuseg \
  --sam_ckpt <initial-checkpoint> --texture --context \
  --use_pms --pms_loss_coef 0.5 --pms_residual_mask_weight 0.3 \
  --pms_preserve_loss_coef 1.0 --pms_preserve_covered \
  --pms_preserve_max_prompts 20 --coverage_accumulate \
  --iterative_baseline_refresh_every 20 \
  --epochs <epochs> --lr <learning-rate> --weight_decay 1e-4 \
  --overlap 92 --b 1
```

Replace angle-bracket placeholders with your paths and training choices. For training from SAM2 initialization, the paper specifies `--pms_start_epoch 50 --lr 1e-4 --lr_min 1e-6`. For a well-performing pretrained initialization, PMS starts immediately. Use `data/tnbc` and `--overlap 32` for TNBC.

`--use_pms` enables self-bootstrapped coverage on the training split. Checkpoints are saved under `logs/<experiment>/Model/`.

## Paper ablations

| Variant | Option |
| --- | --- |
| Without object-score supervision | `--pms_object_weight 0` |
| Without coverage accumulation | `--no-coverage_accumulate` |
| Without preservation prompts | Omit `--pms_preserve_covered` |

For the paper's NMS analysis, evaluate with thresholds 2, 6, 12 and 18. Inference uses the same CA-SAM2 path for all variants.
