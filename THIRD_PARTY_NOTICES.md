# Third-party notices

This repository adapts [CA-SAM2](https://github.com/HanbinHuang123/CA-SAM2). Its original copyright notices are retained. No repository-wide license was present in the upstream CA-SAM2 repository at the time of this release; this project does not grant additional rights to that upstream code.

Included components retain their respective terms:

| Component | License |
| --- | --- |
| [SAM2](https://github.com/facebookresearch/sam2), adapted in `sam2_train/` | [Apache 2.0](docs/licenses/SAM2-Apache-2.0.txt) |
| [timm 0.6.13](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13), ConvNeXt subset | [Apache 2.0](docs/licenses/timm-Apache-2.0.txt) |
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | [MIT](docs/licenses/ConvNeXt-MIT.txt) |
| [OpenMMLab](https://github.com/open-mmlab/mmdetection), adapted FPN | [Apache 2.0](docs/licenses/MMCV-Apache-2.0.txt) |

Local adaptations include the CA-SAM2 feature path, StainPMS supervision, a reduced ConvNeXt package and standalone FPN imports. Dataset and pretrained-weight terms are governed by their original providers.
