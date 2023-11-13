# BTSFusion: Fusion of infrared and visible image via a mechanism of  balancing texture and salience
Yao Qian, Gang Liu∗, Haojie Tang, Mengliang Xing, Rui Chang

Published in: Optics and Lasers in Engineering

- [parer](https://www.sciencedirect.com/science/article/abs/pii/S0143816623004542)

## Abstract
In recent years, deep learning research has received significant attention in the field of infrared and visible  image fusion.  However, the issue of designing loss functions in deep learning-based image fusion methods has  not been well-addressed.  To tackle this problem, we propose a novel mechanism of utilizing traditional fusion  methods as loss functions to guide the training of deep learning models.  We incorporate the superior aspects  of two traditional methods, namely Guided Filter (GF) and Latent Low-Rank Representation (LatLRR), into  the design of the loss function, proposing a fusion method for infrared and visible images that balances both  texture and saliency, termed BTSFusion.  The proposed network is not only lightweight but also preserves the  maximum amount of valuable information in source images.  It is worth noting that the complexity of BTSFusion  primarily lies in the design of the loss function, which allows it to remain an end-to-end network, as demonstrated  by efficiency comparison experiments that highlight the excellent computational efficiency of our algorithm.  Furthermore, through subjective observations and objective comparisons, we validated the performance of the  proposed method by comparing it with twelve state-of-the-art methods on two public datasets.
## Framework
![image](https://github.com/YQ-097/BTSFusion/assets/68978140/4da3f349-0ce2-4070-924b-b41ddfe473bd)

## Recommended Environment

 - [x] pytorch 1.12.1 
 - [x] scipy 1.2.1   
 - [x] numpy 1.11.3

## To Train
The training dataset is temporarily not publicly available. If needed, please contact the author for access.

    python train.py
## To Test
First, parameterize the structure of the trained model, and then run the testing program.

    python net_repvgg.py
    python test1_image.py
## Citation

```
@article{QIAN2024107925,
title = {BTSFusion: Fusion of infrared and visible image via a mechanism of balancing texture and salience},
journal = {Optics and Lasers in Engineering},
volume = {173},
pages = {107925},
year = {2024},
issn = {0143-8166},
doi = {https://doi.org/10.1016/j.optlaseng.2023.107925},
url = {https://www.sciencedirect.com/science/article/pii/S0143816623004542},
author = {Yao Qian and Gang Liu and Haojie Tang and Mengliang Xing and Rui Chang},
keywords = {Image fusion, Guided filter, Latent low-rank representation, Lightweight network, Deep learning},
abstract = {In recent years, deep learning research has received significant attention in the field of infrared and visible image fusion. However, the issue of designing loss functions in deep learning-based image fusion methods has not been well-addressed. To tackle this problem, we propose a novel mechanism of utilizing traditional fusion methods as loss functions to guide the training of deep learning models. We incorporate the superior aspects of two traditional methods, namely Guided Filter (GF) and Latent Low-Rank Representation (LatLRR), into the design of the loss function, proposing a fusion method for infrared and visible images that balances both texture and saliency, termed BTSFusion. The proposed network is not only lightweight but also preserves the maximum amount of valuable information in source images. It is worth noting that the complexity of BTSFusion primarily lies in the design of the loss function, which allows it to remain an end-to-end network, as demonstrated by efficiency comparison experiments that highlight the excellent computational efficiency of our algorithm. Furthermore, through subjective observations and objective comparisons, we validated the performance of the proposed method by comparing it with twelve state-of-the-art methods on two public datasets. The source code will be publicly available at https://github.com/YQ-097/BTSFusion.}
}
```
