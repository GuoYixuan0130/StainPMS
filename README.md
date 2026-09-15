# StainPMS

**StainPMS: Self-Bootstrapped Prompt-Mask Supervision for Nuclei Instance Segmentation**

Yixuan Guo and Meng Zhu · **VCIP 2026**

StainPMS uses accumulated online coverage and residual hematoxylin evidence to supervise the shared mask decoder during training. Inference follows the standard CA-SAM2 path, with no additional learnable parameters.

## Setup

```bash
conda env create -f environment.yml
conda activate stainpms
```

See [Usage](docs/REPRODUCIBILITY.md) for the data layout and training entry point. Datasets and checkpoints are not included.

## Checkpoints

Final models: [MoNuSeg](https://huggingface.co/SetsunaQWQ/StainPMS/resolve/main/StainPMS_monuseg.pth) · [TNBC](https://huggingface.co/SetsunaQWQ/StainPMS/resolve/main/StainPMS_tnbc.pth).

## Evaluate

```bash
python main.py --eval --data_path data/monuseg \
  --sam_ckpt checkpoints/StainPMS_monuseg.pth --texture --context \
  --overlap 92 --test_nms_thr 12 --b 1
```

For TNBC, use `data/tnbc` and `--overlap 32`.

## Paper results

| Dataset | Method | Dice | AJI | DQ | SQ | PQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MoNuSeg | CA-SAM2 | .808 | .644 | .826 | .750 | .620 |
| MoNuSeg | StainPMS | .822 | .666 | .853 | .771 | .658 |
| TNBC | CA-SAM2 | .787 | .639 | .835 | .808 | .676 |
| TNBC | StainPMS | .808 | .665 | .838 | .813 | .682 |

## Citation

```bibtex
@inproceedings{guo2026stainpms,
  title={StainPMS: Self-Bootstrapped Prompt-Mask Supervision for Nuclei Instance Segmentation},
  author={Guo, Yixuan and Zhu, Meng},
  booktitle={IEEE Visual Communications and Image Processing (VCIP)},
  year={2026}
}
```

Built on [CA-SAM2](https://github.com/HanbinHuang123/CA-SAM2) and [SAM2](https://github.com/facebookresearch/sam2). See [third-party notices](THIRD_PARTY_NOTICES.md).
