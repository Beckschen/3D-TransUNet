

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

### Inference & Eval

See scripts/inference.sh


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
