# Spectral super-resolution meets deep learning: achievements and challenges

[Jiang He](http://jianghe96.github.io/), [Qiangqiang Yuan](http://qqyuan.users.sgg.whu.edu.cn/), [Jie Li](http://jli89.users.sgg.whu.edu.cn/), [Yi Xiao](https://xy-boy.github.io/), 
Denghong Liu, [Huanfeng Shen](http://sendimage.whu.edu.cn/shenhf/), 
and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html), Wuhan University
* Codes for the paper entitled as "Spectral super-resolution meets deep learning: achievements and challenges" published in Information fusion. 

* Benchmark about deep learning-based spectral super-resolution algorithms, including the workflows of spectral recovery, colorization, and spectral compressive imaging. 

<div align=center><img src="./supplementary4github/three applications.png" align=center width="720px"/></div>

## Datasets Loading
We give some classical datasets in three applications:

### Spectral recovery: ARAD_1K dataset
The dataset used in spectral recovery is a public hyperspectral image data named ARAD_1K which is released in NTIRE 2022.

We only give the RGB images in github. The label of Training dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7839604). 

Please put the traning dataset into `./2SR/dataset/`, and unzipped it as `.\2SR\dataset\Train_Spec\`.

### Colorization: SUN dataset
We used only part of [SUN dataset](https://cs.brown.edu/~gmpatter/sunattributes.html). Details can be found in our paper. 

We uploaded the test images in github. The training datasets can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7837907). 

Please put the traning dataset into `./1colorization/`, and name it as `color_train.h5`.

### Spectral compressive imaging: CAVE dataset
We designed our SCI procedure following [TSA-Net](https://github.com/mengziyi64/TSA-Net).

The `./3SCI/mask.mat` is the mask used in degradation. Training dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.7839679). 

Please put the traning dataset into `./3SCI/`, and name it as `26train_256_enhanced.h5`.

## Model Zoos
We have collected some classical spectral super-resolution algorithms, including DenseUnet [[42]](https://arxiv.org/pdf/1703.09470.pdf), CanNet [[45]](https://arxiv.org/pdf/1804.04647), HSCNN+ [[50]](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.pdf), sRCNN [[53]](https://www.mdpi.com/2072-4292/11/14/1648/htm), AWAN [[60]](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Li_Adaptive_Weighted_Attention_Network_With_Camera_Spectral_Sensitivity_Prior_for_CVPRW_2020_paper.pdf), FMNet [[69]](https://ojs.aaai.org/index.php/AAAI/article/view/6978/6832), HRNet [[70]](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Zhao_Hierarchical_Regression_Network_for_Spectral_Reconstruction_From_RGB_Images_CVPRW_2020_paper.pdf), HSRnet [[71]](https://ieeexplore.ieee.org/abstract/document/9357488/), HSACS [[73]](https://ieeexplore.ieee.org/abstract/document/9506982), GDNet [[77]](https://ieeexplore.ieee.org/abstract/document/9599509), and SSDCN [[79]](https://ieeexplore.ieee.org/abstract/document/9440658). 

If you want to get the pretrained model, please contact with hej96@gmail.com.

### Your own models
If you want to run your own models with this benchmark, you should put your model `xxxxx.py` into the file `./models/`. And then, you should define your model in specific application. For example, if you want to run your model in spectral recovery, please define your model in `./2SR/model.py`.

## Running Details
The code has been tested on PyTorch 1.6.

### Spectral recovery
We improved our implementation inspired by [MST++](https://github.com/caiyuanhao1998/MST-plus-plus).

For training, you should run `train.py` at the path `.\2SR`:

    python train_adalr.py --method HSRnet --batchSize 2 --gpus 0

More details can be found in the help of 'argparse' in `train.py`.


After training, you can run `test.py` to obtain the testing results:

    python demo_try.py -- model CanNet --name CanNet_b8_adalr0.0008 --time 2023_04_17_23_09_36 --epoch 200 --gpus 0 --data_root dataset/

'time' and 'name' can be found in the `.\2SR\checkpoint\`.

### Colorization
For training, you should run `train_adalr.py` at the path `.\1colorization`:

    python train_adalr.py --method HSRnet_color --batchSize 2 --gpus 0

More details can be found in the help of 'argparse' in `train_adalr.py`.


After training, you can run `demo_try.py` to obtain the testing results:

    python demo_try.py --name CanNet_10_b8_adam_L1loss_adalr0.0004 --scale 1 --gpus 0

'scale' is used to calculate ERGAS, which is the spatial resolution ratio. Notice: you should change the 'bestepoch' in `demo_try.py` or just change 'path' with your checkpoint path.


### Spectral compressive imaging
For the common training, you should run `train.py` at the path `.\3SCI`:

    python train.py --method SSDCN --batchSize 2 --gpus 0

More details can be found in the help of 'argparse' in `train.py`.


For the new assumptive spectral imaging in our paper, you should run `train_meaninput.py` at the path `.\3SCI`:

    python train_meaninput.py --method CanNet --batchSize 2 --gpus 0`. 
    
More details can be found in the help of 'argparse' in `train_meaninput.py`. Before training, you should download the new training data set `26train_256_enhanced.h5`.


After training, you can run `test.py` to obtain the testing results:

    python test.py --name FMNet_step10_b1_adam_L1loss_adalr0.0001 --scale 1 --gpus 0

'scale' is used to calculate ERGAS, which is the spatial resolution ratio. Notice: the 'test_ite ' is chosen as 200 in this application.

## Contact

If any questions, please feel free to contact with [hej96.work@gmail.com].

## Reference
Please cite: 
```
@article{hj2023_DL4sSR,
title={Spectral super-resolution meets deep learning: achievements and challenges},
author={He, Jiang and Yuan, Qiangqiang and Li, Jie and Xiao, Yi and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},
journal={Information Fusion},
volume={97},
pages={101812},
year={2023},
}
```
