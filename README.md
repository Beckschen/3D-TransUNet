

This is the official repository of our project ["3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers"](https://arxiv.org/abs/2310.07781). 

## 📰 News

- [7/26/2024] TransUNet, which supports both 2D and 3D data and incorporates a Transformer encoder and decoder, has been featured in the journal Medical Image Analysis ([link](https://www.sciencedirect.com/science/article/pii/S1361841524002056)).
```bibtex
@article{chen2024transunet,
  title={TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and others},
  journal={Medical Image Analysis},
  pages={103280},
  year={2024},
  publisher={Elsevier}
}
```
## Usage

### Installation

See scripts/install.sh for installation. See [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for self-configuring data preprocessing.

### Train

See scripts/train.sh

### Inference & Test

See scripts/inference.sh

### 5 fold Eval

See scripts/eval.sh

We provide 5-fold model checkpoints for the Brats and HepaticVessel datasets. These models can be directly downloaded from HuggingFace.
```
huggingface-cli download qicq1c/3D-TransUNet --local-dir .
```

5-fold results for the Brats dataset

| Fold | ncr | ed | et | Avg |
|------|----------|----------|----------|----------|
| 1    | 0.9348   | 0.9130   | 0.8590   | 0.9023   |
| 2    | 0.9327   | 0.9068   | 0.8639   | 0.9011   |
| 3    | 0.9389   | 0.9245   | 0.8752   | 0.9129   |
| 4    | 0.9394   | 0.9254   | 0.8773   | 0.9140   |
| 5    | 0.9418   | 0.9141   | 0.8560   | 0.9040   |
| **Avg** | **0.9371** | **0.9168** | **0.8663** | **0.9061** |

5-fold results for the HepaticVessel dataset

| Fold | Vessel | Tumour | Avg |
|------|----------|----------|----------|
| 1    | 0.6691   | 0.7153   | 0.6922   |
| 2    | 0.6393   | 0.6899   | 0.6646   |
| 3    | 0.6215   | 0.7276   | 0.6745   |
| 4    | 0.6684   | 0.6735   | 0.6710   |
| 5    | 0.6110   | 0.6745   | 0.6428   |
| **Avg** | **0.6428** | **0.6966** | **0.6690** |


## Acknowledgements

This work is partially supported by TPU Research Cloud program, Google Cloud Research Credits program, and AWS Cloud Credit for Research program. Thanks for the codebase from [Mask2former](https://github.com/facebookresearch/Mask2Former), [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [TransUNet](https://github.com/Beckschen/TransUNet)


If you find 3D-TransUNet useful for your research and applications, please cite using this BibTeX:

```
@article{chen2023transunet3d,
  title={3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers},
  author={Chen, Jieneng and Mei, Jieru and Li, Xianhang and Lu, Yongyi and Yu, Qihang and Wei, Qingyue and Luo, Xiangde and Xie, Yutong and Adeli, Ehsan and Wang, Yan and Lungren, Matthew and Xing, Lei and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2310.07781},
  year={2023}
}
```
