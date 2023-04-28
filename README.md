# DL4sSR
* Codes for the paper entitled as "Spectral super-resolution meets deep learning: achievements and challenges" published in Information fusion. 

* Benchmark about deep learning-based spectral super-resolution algorithms, including the workflows of spectral recovery, colorization, and spectral compressive imaging. 

<div align=center><img src="./three" align=center width="360px"/></div>






# Reference
Please cite: 

@article{hj2023_DL4sSR,
title={Spectral super-resolution meets deep learning: achievements and challenges},
author={He, Jiang and Yuan, Qiangqiang and Li, Jie and Xiao, Yi and Liu, Denghong and Shen, Huanfeng and Zhang, Liangpei},
journal={Information Fusion},
volume={},
pages={},
year={2023},
}


# Reference
Zhang, Q., Yuan, Q., Li, J., Wang, Y., Sun, F., and Zhang, L.: Generating seamless global daily AMSR2 soil moisture (SGD-SM) long-term products for the years 2013¨C2019, Earth Syst. Sci. Data, 13, 1385¨C1401, https://doi.org/10.5194/essd-13-1385-2021, 2021.

# Dataset Download

* **Link 1**: [Baidu Yun](https://pan.baidu.com/s/1SGdKmfgUgUBmcWse-cDsWg) (Extracting Code: fu8f)

* **Link 2**: [Google Drive](https://drive.google.com/file/d/1pGoX12Va3k6o9ybIMBjpDDHLbcUShM1P/view?usp=sharing)

* **Link 3**: [Zenodo](http://doi.org/10.5281/zenodo.4417458)


# Environments and Dependencies
* Windows 10
* Python 3.7.4
* netCDF4
* numpy


# Toolkit Installation
This soil moisture dataset is comprised of netCDF4 (\*.nc) files. Therefore, users need to install netCDF4 toolkit before reading the data:
```
    pip install netCDF4
    pip install numpy
```

# Data Reading
It should be noted that the original and reconstructed soil moisture data are both recorded in a NC file. 
User can read the original data, reconstructed data, and mask data as follows (more details can be viewed in [Example.py](Example.py)):
```
    Data = nc.Dataset(NC_file_position)
    Ori_data = Data.variables['original_sm_c1']
    Rec_data = Data.variables['reconstructed_sm_c1']
    Ori = Ori_data[0:720, 0:1440]
    Rec = Rec_data[0:720, 0:1440]
    Mask_ori = np.ma.getmask(Ori)
```



# Data Visualization
Users can visualize \*.nc format file through [Panoply](https://www.giss.nasa.gov/tools/panoply/download/) software. Before visualizing, you must install [Java SE Development Kit](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).


# Contact Information
If you have any query for this work, please directly contact me.

Author: Qiang Zhang, Wuhan Unviversity.

E-mail: whuqzhang@gmail.com

Homepage: [qzhang95.github.io](https://qzhang95.github.io/)