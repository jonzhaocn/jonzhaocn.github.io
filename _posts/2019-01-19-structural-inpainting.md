---
layout: article
title:  "structural inpainting tensorflow实现"
date:   2019-01-19 10:43:57 +0800
categories: [blog]
tags: [tensorflow, inpainting, deep-learning]
mathjax: false
---

### 1 概述

`structural inpainting`，作者在`context encoder[2]`的基础上进行改进，在网络中加入了`feature reconstruction loss`，`feature reconstruction loss`与`MSE loss`的线性组合构成了`structural loss`，以此来提升修复区域中结构信息的修复效果。
* 😛😜😝代码托管在[github](https://github.com/jonzhaocn/structural_inpainting)



### 2 网络结构与损失函数

`structural inpainting`使用到了3种损失，`feature reconstruction loss`、`MSE loss`与`adversarial loss`，其中`MSE loss`与`adversarial loss`的使用来源于`context encoder`。在训练过程中，`context encoder`与鉴别网络构成了对抗关系，`context encoder`用来对缺失信息的图像进行修复，而鉴别网络用来对图像进行分类，判断图像是`context encoder`修复出来的图像，还是`ground truth`。而对于`feature reconstruction loss`，则需要使用到`VGG16`提取图像的特征来计算。
#### 2.1 网络结构
如图`1`所示，整个网络结构由三部分组成，`context encoder`，鉴别网络与`VGG16`，其中`context encoder`和鉴别网络是需要训练的部分，而`VGG16`在训练过程其权重值不变，使用`VGG16`来提取图像的特征，以计算`feature reconstruction loss`。图中，$X$是待修复图像，$y$为`context encoder`的输出，包含了修复信息，$\hat{X}$为`ground truth`，$\hat{X}_C$为`ground truth`对应的鉴别网络输入，$D_{W'}$是鉴别网络。

$X$经过`context encoder`修复之后，得到$y$，`ground truth`$\hat{X}$截取中心部分的信息之后得到$\hat{X}_C$，$y$与$\hat{X}_C$作为鉴别网络输入，让鉴别网络进行分类，计算得到`adversarial loss`。$y$与$\hat{X}_C$进入`VGG16`分别提取特征，计算两张图像特征之间的差别，得到`feature reconstruction loss`。

![图1 网络结构 来源：[1]论文](/imgs/structural-inpainting/network-structure.png)

#### 2.2 feature reconstruction loss
`feature reconstruction loss`的思想来源于论文`[3]`提出的`perceptual loss`，`perceptual loss`由`feature loss`和`style loss`组成，其中的一个要点就是使用`VGG16`来提取生成图像与`ground truth`的特征，比较两张图像在特征之间的差别，以此来指导网络权重的迭代调整。而之前，比较生成图像与`ground truth`之间的差别，多数情况下使用的是`MSE loss`，就是比较图像像素级别的差距。

`[1]`作者认为`adversarial loss`的使用，有利于修复缺失区域的纹理，但是对于修复区域的结构信息贡献较小，所以在网络训练中加入了`feature reconstruction loss`，想要以此提升网络对图像结构的修复质量，并将`MSE loss`与`feature reconstruction loss`的线性组合成为了`structural loss`。

`feature reconstruction loss`的加入有提升网络对于结构信息的修复质量，这一点可以从图`2`中看出，图`2`对比了`patch-based`、`context encoder`与加了`feature reconstruction loss`的`context encoder`，可以看到，最右边一排的图像对比`context encoder`，在图像结构方面的修复质量更好。
![图2 来源：[1]论文](/imgs/structural-inpainting/2.png)

#### 2.3 使用VGG16提取哪些特征
如图`3`所示，对比了在`VGG16`中不同的网络层中计算图像的特征之间的差距对于最终修复效果的影响，发现使用`MSE loss`与`VGG16`的`conv1_1`、`conv2_1`、`conv3_1`的组合的效果最好，即需要对比两张图像在像素级别的差距，再使用`VGG16`提取它们在`conv1_1`、`conv2_1`、`conv3_1`层的特征并计算`feature reconstruction loss`。
![图3 来源：[1]论文](/imgs/structural-inpainting/3.png)



### 3 参考文献
1. `Vo H V, Duong N Q K, Perez P. Structural inpainting[J]. arXiv preprint arXiv:1803.10348, 2018.`
2. . `Pathak D, Krahenbuhl P, Donahue J, et al. Context encoders: Feature learning by inpainting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2536-2544.`
3. . `Johnson J, Alahi A, Fei-Fei L. Perceptual losses for real-time style transfer and super-resolution[C]//European Conference on Computer Vision. Springer, Cham, 2016: 694-711.`