---
layout: article
title:  "GAN网络matlab实现"
date:   2018-06-01 14:17:26 +0800
categories: [blog]
tags: [matlab, gan]
mathjax: false
---

### 1 概述

此代码是在`matlab`平台上搭建了一个简单的`toolbox`，用来实现生成对抗性网络（`Generative Adversarial Net`），在代码中加入了卷积、反卷积、扩张卷积等网络层的实现。

网络层有卷积(`conv2d`)，反卷积(`conv2d transpose`)，扩张卷积(`atrous conv2d`)，下采样(`sub sampling`)，全连接(`fully connect`)和`reshape`，激活函数支持`tanh`、`sigmoid`、`relu`、`leaky_relu`，不过discriminator的最后一层的激活函数仅支持`sigmoid`。

由于我实现反卷积、扩张卷积的操作是通过在输入或卷积核中插入0值实现的，这样是一种低效的实现方式，代码的效率不高，但是可以作为`matlab`搭建神经网络的一些参考。

如果需要在`matlab`上高效地搭建卷积神经网络，可以使用[matconvnet](http://www.vlfeat.org/matconvnet/)。`github`上有一个使用`matconvnet`搭建`dcgan`的例子：[mcnDCGAN](https://github.com/hbilen/mcnDCGAN)
#### 1.1 更多
* [MNIST数据集下载](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat)

* 😛😜😝代码托管在github上：[GAN-Base-on-Matlab](https://github.com/jonzhaocn/GAN-Base-on-Matlab)



### 2 代码说明

#### 2.1 代码的文件夹结构

```
activation/    激活函数
error_term/    计算残差
gradient/    计算梯度
layer/    各种卷积操作、reshape和全连接
test/    测试使用
util/    提供基本操作的函数
gan_train.m 训练函数

nerual_network_flow文件夹中：
nerual_network_flow/nn_applygrads_adam.m    使用adam算法更新网络
nerual_network_flow/nn_applygrads_sgd.m    使用sgd算法更新网络
nerual_network_flow/nn_bp_d.m    discriminator的bp函数
nerual_network_flow/nn_bp_g.m    generator的bp函数
nerual_network_flow/nn_ff.m    前向传播函数
nerual_network_flow/nn_setup.m    网络创建函数
```


#### 2.2 网络层的输入输出size

* `conv2d`、`conv2d_transpose`、`atrous_conv2d`的卷积输入需要是一个四维数组`[image_height, image_width, image_channel, batch_size]`。
* `fully_connect`的输入和输出都是一个二维数组，`[data, batch_size]`。



#### 2.3 搭建一个网络层时需要使用到的参数

```
type    网络层的类型
output_shape    网络层的输出size
activation    所指定的激活函数：sigmoid，relu，tanh、leaky_relu
output_maps    输出的特征图个数
kernel_size    卷积核大小
stride    步长，conv2d层只支持步长为1的卷积操作，需要步长为2的话，就要配合下采样层一起使用
padding    卷积操作指定的padding模式，支持same或者valid
rate    扩张卷积的扩张率
scale    下采样层的缩小比率
```


#### 2.4 使用struct表示神经网络的结构

```
输入层
struct('type', 'input', 'output_shape', [batch_size, 100])
全连接层
struct('type', 'fully_connect', 'output_shape', [batch_size, 3136], 'activation', 'leaky_relu')
反卷积层
struct('type', 'conv2d_transpose', 'output_shape', [batch_size, 14, 14, 10], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'leaky_relu')
卷积层
struct('type', 'conv2d', 'output_maps', 10, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
扩张卷积
struct('type', 'atrous_conv2d', 'output_maps', 10, 'rate', 2, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
下采样
struct('type', 'sub_sampling', 'scale', 2)
```


### 3 使用示例

#### 3.1 example_1
`example_1`中搭建了一个简单的GAN的网络结构，用来生成手写数字图片，使用`matlab`结构体设置好生成器和鉴别器的网络结构之后，可以通过`gan_train`函数对网络进行训练。
`example_1` 训练需要较长的训练时间，如果想快速看到训练结果的，可以尝试简化`generator`和`discriminator`的层数、网络节点数。
```matlab
clc;
% -----------load mnist data
load('mnist_uint8', 'train_x');
train_x = double(reshape(train_x, 60000, 28, 28))/255;
% train_x:[height, width, channel, images_index]
train_x = permute(train_x,[3,2,4,1]);
batch_size = 64;
% ----------- model
generator.layers = {
    struct('type', 'input', 'output_shape', [100, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [3136, batch_size], 'activation', 'leaky_relu')
    struct('type', 'reshape', 'output_shape', [7,7,64, batch_size])
    struct('type', 'conv2d_transpose', 'output_shape', [14, 14, 32, batch_size], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'conv2d_transpose', 'output_shape', [28, 28, 1, batch_size], 'kernel_size', 5, 'stride', 2, 'padding', 'same', 'activation', 'sigmoid')
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [28, 28, 1, batch_size])
    struct('type', 'conv2d', 'output_maps', 32, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'conv2d', 'output_maps', 64, 'kernel_size', 5, 'padding', 'same', 'activation', 'leaky_relu')
    struct('type', 'sub_sampling', 'scale', 2)
    struct('type', 'reshape', 'output_shape', [3136, batch_size])
    struct('type', 'fully_connect', 'output_shape', [1, batch_size], 'activation', 'sigmoid')
};
args = struct('batch_size', batch_size, 'epoch', 10, 'learning_rate', 0.001, 'optimizer', 'adam');
[generator, discriminator] = gan_train(generator, discriminator, train_x, args);
```
`example_1`的结果：
![epoch_5_t_500.png](/assets/gan-base-on-matlab/example1-result1.png)

![epoch_6_t_500.png](/assets/gan-base-on-matlab/example1-result2.png)



#### 3.2 example_2

`example_2`同样的也是使用了手写数字作为训练数据集，训练GAN来生成手写数字图片。

```matlab
generator.layers = {
    struct('type', 'input', 'output_shape', [100, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [1024, batch_size], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [28*28, batch_size], 'activation', 'sigmoid') 
    struct('type', 'reshape', 'output_shape', [28, 28, 1, batch_size])
};
discriminator.layers = {
    struct('type', 'input', 'output_shape', [28,28,1, batch_size])
    struct('type', 'reshape', 'output_shape', [28*28, batch_size]) 
    struct('type', 'fully_connect', 'output_shape', [1024, batch_size], 'activation', 'relu')
    struct('type', 'fully_connect', 'output_shape', [1, batch_size], 'activation', 'sigmoid') 
};
```
`example_2`的结果
![epoch_7_t_1000.png](/assets/gan-base-on-matlab/example2-result1.png)



### 4 参考文献

1. `https://grzegorzgwardys.wordpress.com/2016/04/22/8/`
2. `Dumoulin V, Visin F. A guide to convolution arithmetic for deep learning[J]. arXiv preprint arXiv:1603.07285, 2016.`
3. `https://github.com/rasmusbergpalm/DeepLearnToolbox/tree/master/CNN`
4. `http://neuralnetworksanddeeplearning.com/index.html`