---
layout: article
title:  exemplar-based image inpainting matlab
date:   2018-08-11 20:02:58 +0800
categories: [blog]
tags: [matlab, inpainting]
mathjax: true
---

### 1 概述

本代码是对`Region filling and object removal by exemplar-based image inpainting`的`MATLAB`实现，用来对图像进行区域填充、物体移除。

😛😜😝代码托管在`github`上：[exemplar-based-image-inpainting](https://github.com/jonzhaocn/exemplar-based-image-inpainting)
### 2 算法
如图1所示，为算法伪代码。
![图1 算法伪代码](/imgs/exemplar-based-image-inpainting/pseudo-code.png)

#### 2.1 区域划分
实现这个算法，首先需要进行图像区域的划分。如图2所示，$\Omega$为目标区域，是需要进行填充的区域，$\phi$为源区域，作为填充数据的来源。$\Omega$的边界为$\delta\Omega$。如果从像素值来看的话，可以将目标区域的像素值设置为`0`，但是`0`在图像中为黑色的意思，所以最好可以有一个辅助数组来标志缺失区域。
![图2 区域划分与边界](/imgs/exemplar-based-image-inpainting/boundary.png)

#### 2.2 计算边界区域
使用下面的代码可以方便地计算出边界点的位置。`map`是一个二值数组，只有0、1，其中1表示缺失像素值所在位置。计算出来的结果`result`中，1所在的位置就是边界点所在的位置。
```matlab
result = imdilate(map, se) - map;
```
#### 2.3 优先级的计算公式
边界上有很多像素点，以这些点为中心可以得到很多patch（比如9×9的patch），对于这些patch，都需要计算一下他们的优先级是多少，以便从中选出一个优先级最大的块作为首要修复的对象。

* 对于一个patch，块的优先级计算公式：
$P(p)=C(p) \ast D(p) \tag{1}$
其中，$p$代表这个待修复块的中心点，$C(p)$表示块的置信度，$D(p)$为data term。

* 置信度的计算公式是：
$C(p)=\frac{\sum_{q\in\psi_p\cap(I-\Omega)} c(q)}{\psi_p} \tag{2}$
其中，$C(p)$表示置信度，其中$I$为整张图像，$\Omega$为$I$的缺失区域，$\psi_p$为待填充的块，$|\psi_p|$为块的面积。$c(q)$为块中的像素点的置信度，在初始化的时候，已存在的像素点的置信度为`1`，缺失的像素点的置信度为`0`，在修复过程中，修复出来的像素点的置信度被更新为块的置信度$C(p)$，由此可知，随着修复过程的推进，修复出来的像素点的置信度会越来越小。

* data term的计算公式为：
$D(p) =\frac{|\bigtriangledown I^{\perp}_p \ast n_p|}{\alpha} \tag{3}$
公式3，这里是等照度向量与法向量$n_p$的点乘再求模长，$\alpha$为归一化因子。
在计算等照度向量的时候需要先计算$p$点的`image gradient`，等照度线向量为`gradient`逆时针旋转90°，`gradient`代表了像素值变化最快的方向，而等照度线向量与`gradient`垂直，代表了变化最慢的方向。
* image gradient
$\bigtriangledown f = [ \begin{matrix} g_x \\ g_y \end{matrix}]=[ \begin{matrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{matrix}] \tag{4}$
#### 2.4 暴力搜索
计算出每一个patch的优先级之后，从中选出一个优先级最高的patch作为待修复对象，如$Pt$。再使用暴力法，从源区域中选出一个与之最相近的块$Ps$，将$Ps$中对应的像素拷到$Pt$中（这里只需要修复$Pt$中缺失的像素点）。使用暴力法的时候，衡量两个块之间的距离时，使用`SSD`（差的平方和）作为距离，由于$Pt$本身就缺失了部分像素值，所以计算的就是$Pt$非缺失像素和其他patch对应位置像素值的`SSD`。

注意，在修复过程的迭代中，缺失区域逐渐变小，但是提供像素来源的目标区域固定不变。因为目标区域中，修复出来的像素值可信度比较低，不予以采用。
### 3 修复结果
![原图](/imgs/exemplar-based-image-inpainting/original-image.png)
![缺失信息](/imgs/exemplar-based-image-inpainting/masked-image.png)
![修复之后](/imgs/exemplar-based-image-inpainting/inpainted-image.png)

### 4 参考文献
1. `Criminisi A, Pérez P, Toyama K. Region filling and object removal by exemplar-based image inpainting[J]. IEEE Transactions on image processing, 2004, 13(9): 1200-1212.`
2. `Criminisi A, Perez P, Toyama K. Object removal by exemplar-based inpainting[C]//Computer Vision and Pattern Recognition, 2003. Proceedings. 2003 IEEE Computer Society Conference on. IEEE, 2003, 2: II-II.`
3. `https://github.com/IouJenLiu/Region-Filling-and-Object-Removal`