# Spectral super-resolution meets deep learning: achievements and challenges

[Jiang He](http://jianghe96.github.io/), [Qiangqiang Yuan](http://qqyuan.users.sgg.whu.edu.cn/), [Jie Li](http://jli89.users.sgg.whu.edu.cn/), [Yi Xiao](https://xy-boy.github.io/), 
Denghong Liu, [Huanfeng Shen](http://sendimage.whu.edu.cn/shenhf/), 
and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html), Wuhan University
* Codes for the paper entitled as "Spectral super-resolution meets deep learning: achievements and challenges" published in Information fusion. 

* Benchmark about deep learning-based spectral super-resolution algorithms, including the workflows of spectral recovery, colorization, and spectral compressive imaging. 

<div align=center><img src="./supplementary4github/three applications.png" align=center width="720px"/></div>

## Datasets Loading
We give some classical datasets in three applications:

### Spectral Recovery: ARAD_1K
The dataset used in spectral recovery is a public hyperspectral image data named ARAD_1K which is released in NTIRE 2022.

We only give the RGB images in github. The label of Training dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7839604). 

Please put the traning dataset into `./2SR/dataset/`, and unzipped it as `.\2SR\dataset\Train_Spec\`.

### Colorization: SUN



### Spectral Compressive Imaging: CAVE
We designed our SCI procedure following [TSA-Net](https://github.com/mengziyi64/TSA-Net).

The `./3SCI/mask.mat` is the mask used in degradation. Training dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7839679). 

Please put the traning dataset into `./3SCI/`, and name it as `26train_256_enhanced.h5`.


## Model Zoos

## Running
The code has been tested on PyTorch 1.6.
Before running, you should put your model `xxxxx.py` into the file `models`
## Spectral Recovery




# Reference
Please cite: 
```
@article{hj2023_DL4sSR,
title={Spectral super-resolution meets deep learning: achievements and challenges},
author={He, Jiang and Yuan, Qiangqiang and Li, Jie and Xiao, Yi and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},
journal={Information Fusion},
volume={},
pages={},
year={2023},
}
```