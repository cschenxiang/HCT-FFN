# Hybrid CNN-Transformer Feature Fusion for Single Image Deraining

Xiang Chen, Jinshan Pan, Jiyang Lu, Zhentao Fan, Hao Li

<hr />

> **Abstract:** *Since rain streaks exhibit diverse geometric appearances and irregular overlapped phenomena, these complex characteristics challenge the design of an effective single image deraining model. To this end, rich local-global information representations are increasingly indispensable for better satisfying rain removal.  In this paper, we propose a lightweight Hybrid CNN-Transformer Feature Fusion Network (dubbed as HCT-FFN) in a stage-by-stage progressive manner, which can harmonize these two architectures to help image restoration by leveraging their individual learning strengths. Specifically, we stack a sequence of the degradation-aware mixture of experts (DaMoE) modules in the CNN-based stage, where appropriate local experts adaptively enable the model to emphasize spatially-varying rain distribution features. As for the Transformer-based stage, a background-aware vision Transformer (BaViT) module is employed to complement spatially-long feature dependencies of images, so as to achieve global texture recovery while preserving the required structure.  Considering the indeterminate knowledge discrepancy among CNN features and Transformer features, we introduce an interactive fusion branch at adjacent stages to further facilitate the reconstruction of high-quality deraining results. Extensive evaluations show the effectiveness and extensibility of our developed HCT-FFN.* 
<hr />

## Network Architecture

<img src = "figure/network.png"> 

## Installation
* PyTorch == 0.4.1
* torchvision0.2.0
* Python3.6.0
* imageio2.5.0
* numpy1.14.0
* opencv-python
* scikit-image0.13.0
* tqdm4.32.2
* scipy1.2.1
* matplotlib3.1.1
* ipython7.6.1
* h5py2.10.0

## Citation
If you are interested in this work, please consider citing:

    @inproceedings{chen2023hybrid,
        title={Hybrid CNN-Transformer Feature Fusion for Single Image Deraining}, 
        author={Chen, Xiang and Pan, Jinshan and Lu, Jiyang and Fan, Zhentao and Li, Hao},
        booktitle={AAAI},
        year={2023}
    }

## Acknowledgment
This code is based on the [SPDNet](https://github.com/Joyies/SPDNet). Thanks for sharing !
