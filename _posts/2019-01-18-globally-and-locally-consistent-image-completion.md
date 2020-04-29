---
layout: article
title:  "globally and locally consistent image completion tensorflow实现"
date:   2019-01-18 19:40:33 +0800
categories: [blog]
tags: [tensorflow, inpainting, deep-learning]
mathjax: true
---

### 1 概述

在`globally and locally consisten image completion`中，作者在`context encoder`的基础上进行改进，网络由一个修复网络和两个鉴别网络组成。使用两个鉴别网络来对图像进行分类，判断图像是修复出来的图像，还是`ground truth`。

* 😛😜😝代码托管在[github](https://github.com/jonzhaocn/globally-and-locally-consistent-image-completion-tensorflow)



### 2 网络结构

#### 2.1 网络结构综述
网络结构如图`1`所示，整个网络由一个修复网络和两个鉴别网络（全局鉴别网络和局部鉴别网络）组成。修复网络用来对进入其中的图像进行修复，而鉴别网络用来对图像进行判别。鉴别网络的目标是尽可能准确地分类修复出来的图像和`ground truth`，而修复网络则是尽可能地去愚弄鉴别网络，即要提升修复质量，使得鉴别网络无法准确地分辨修复出来地图像与`ground truth`。修复网络与鉴别网络组成了生成对抗性网络，以此来提高图像修复质量。

一张`ground truth`，使用随机生成的`mask`遮盖掉其中的一部分信息，产生待修复图像。待修复图像进入修复网络中，得到修复后的图像。全局鉴别网络输入整张修复后的图像，而局部鉴别网络输入图像的一个局部区域，局部区域中包含了修复出来的图像信息，两个鉴别网络的输出都是一个`1x1024`的向量，将两个`1x1024`拼接成`1x2048`的向量，在经过一个全连接层，得到最终的值，即为鉴别网络对图像的分类结果（`real or fake`），所以两个鉴别网络是一起工作的。

训练时，需要使用到修复网络与鉴别网络，训练完成之后，就只需要使用到修复网络了。
![图1 网络结构 来源：[1]论文](/imgs/globally-and-locally-consistent-image-completion/network-structure.png)

#### 2.2 修复网络可以修复任意分辨率的图像吗
修复网络是一个全卷积网络，因此训练完成之后的修复网络可以用来修复任意分辨率的图像。当然，测试图像的分辨率如果与训练时的分辨率有较大差别的时候，修复效果比较差。所以这里说的修复任意分辨率的图像，更多的是想说，训练完成后，修复网络可以输入任意分辨率的图像，但是不能保证对任意分辨率的图像的修复效果都比较好。
#### 2.3 训练时，鉴别网络的输入是什么
全局鉴别网络与局部鉴别网络分别关注待判别图像的不同部分。全局鉴别网络的输入是整张图像，而局部鉴别网络的输入是图像中一个局部区域，如果这张图像是修复出来的图像，则该局部区域包含修复出来的区域，如果这张图像是`ground truth`则该局部区域就是图像中的任意一个区域。`ground truth`中并不包含修复出来的信息，但是还是需要指定一个区域作为局部鉴别网络的输入，因为鉴别网络在没有图像输入之前，还没有对图像的分类结果产生。如果`ground truth`与修复出来的图像像鉴别网络中输入的模式不一样，一个需要向局部鉴别网络输入信息，而另外一个不需要的话，那么鉴别网络就很容易将图像划分成两类了。
#### 2.4 为什么需要使用到两个鉴别网络
![图2 鉴别网络的作用 来源：[1]论文](/imgs/globally-and-locally-consistent-image-completion/discriminator-effect.png)
图`2`中，`(a)`为待修复图像，`(b)`中只使用到了`MSE loss`，(c)中使用到了`MSE loss`与全局鉴别网络，`(d)`中使用到了`MSE loss`与局部鉴别网络，`(e)`中使用到了`MSE loss`与两个鉴别网络。

如图`2`所示，`(b)(c)`中没有使用到局部鉴别网络，导致修复区域的内容会比较模糊，而`(d)`中局部鉴别网络的使用，增加了修复区域的细节，但是没有使用全局鉴别网络，修复区域缺乏全局的一致性。



### 3. 网络训练
#### 3.1 MSE loss
$L(x, M_c)=\| M_c \odot (C(x,M_c)-x)\|^2 \tag{1}$
其中，$\odot $表示矩阵点乘，$\| \|$表示欧几里得距离。
#### 3.2 GAN loss
$\min\limits_{C}\ \max\limits_{D}\ E[log D(x, M_d)+log(1-D(C(x,M_c),M_c))] \tag{2}$
其中，$C$代表修复网络，$D$代表鉴别网络，修复网络与鉴别网络形成了对抗性关系。$M_c$有两个作用，在图像进入修复网络之前，使`ground truth`缺失了部分信息；在图像修复完成之后，用来指示包含修复信息的局部区域作为局部鉴别网络的输入。$M_d$是随机生成的mask，但是它的作用只是用来指示`ground truth`中的某个区域作为局部鉴别网络的输入，并没有使`ground truth`缺失信息。
#### 3.3 联合损失
$\min\limits_{C}\ \max\limits_{D}\ E[L(x, M_c)+\alpha logD(x,M_d)+\alpha log(1-D(C(x,M_c),M_c))] \tag{3}$
这里，最重要的是$\alpha$，它用来调节`GAN loss`在联合损失中的比重，在文中作者将$\alpha$设置成了`0.0004`。$\alpha$过小，`GAN loss`就发挥不出作用，修复出来的图像比较平滑，缺失细节，如果$\alpha$过大，训练过程不稳定，修复效果也不好。
#### 3.4 训练
网络的训练分为三个部分：
* 第一个部分，使用MSE loss（公式1）单独训练修复网络，不更新鉴别网络
* 第二个部分，使用GAN loss（公式2），单独训练鉴别网络，不更新修复网络
* 第三个部分，使用联合损失，结合修复网络与鉴别网络一起训练，但是根据生产对抗性网络的做法，两个网络是轮流更新的
![图3 训练算法 来源：[1]论文](/imgs/globally-and-locally-consistent-image-completion/training-pseudo-code.png)



### 4 参考文献
1. `Iizuka S, Simo-Serra E, Ishikawa H. Globally and locally consistent image completion[J]. ACM Transactions on Graphics (TOG), 2017, 36(4): 107.` 

2. `Pathak D, Krahenbuhl P, Donahue J, et al. Context encoders: Feature learning by inpainting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2536-2544.`