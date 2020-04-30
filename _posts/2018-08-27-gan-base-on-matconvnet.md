---
layout: article
title:  "使用 MatConvNet 搭建GAN网络"
date:   2018-08-27 20:07:21 +0800
categories: [blog]
tags: [matlab, gan]
mathjax: false
---

### 1 概述

该代码使用`MatConvNet`在`matlab`上搭建`GAN`网络，用来生成手写数字图片。

`MatConvNet`是一个开源的、用来在`matlab`上搭建高效卷积神经网络的`toolbox`。在这里，使用`MatConvNet`搭建一个简单的`GAN`网络来生成手写数字图像，为`MatConvNet`的入门提供一个例子。

* `keyword`：`matlab`，`GAN`，`Generative Adversarial Nets`， 生成对抗性网络

<!--more-->

#### 1.1 More
* 😛😜😝代码托管在[github](https://github.com/jonzhaocn/GAN_base_on_matconvnet)



### 2 安装
首先需要到 `MatConvNet `的官网下载 `MatConvNet `的源码，接下在本机中编译源码，在`Windows` 编译源码时需要使用到 `Visual Studio`。编译 `MatConvNet` 时，在` MatContNet` 的命令行窗口中使用`vl_compilenn()`函数进行编译。如果要编译成支持 `gpu `的版本，就需要先下载好 `CUDA`。如果需要使用到 `cudnn`库来
加速，则需要提前下载好 `CUDA` 和 `cudnn`。

* [MatConvNet官网](http://www.vlfeat.org/matconvnet/)
* [各个Matlab版本对应的编译器版本](https://ww2.mathworks.cn/support/sysreq/previous_releases.html)
* [MatConvNet安装编译指导](http://www.vlfeat.org/matconvnet/install/)



### 3 搭建网络
#### 3.1 确定网络结构
首先需要确定训练数据的`size`，一张手写数字图像的`size`是`28×28×1`，是灰度图，所以只有一个通道。`GAN`网络有一个生成器和一个鉴别器，将随机生成的噪声通过生成器得到一张假的手写数字图像。真实图像和假图像经过鉴别器得到分别得到`1×1`的标签值。随机噪声的`size`可以自由确定，在这里我将它设置成`1×1×100`的高斯噪声，其实就是只有`100`个点，将它设置成`100`个`channel`方便在生成器中进行操作。

到这里，网络的输入输出就确定了，向生成器中输入一个`1×1×100`的噪声得到一张`28×28×1`的图像。向鉴别器中输入一张`28×28×1`的图像得到一个`1×1×1`的`label`（概率值）。

接下来就可以确定网络的结构，鉴别器和生成器的具体结构如表1、2所示。表中给出了网络中每一层的类型、步长、`padding`等。在这里确定网络结构的标准就是保证的网络输入输出的`size`与预设的一致即可。
![](/imgs/gan-base-on-matconvnet/discriminator-structure.png)

![](/imgs/gan-base-on-matconvnet/generator-structure.png)

#### 3.2 如何向网络中添加一个网络层
表1、2中的`layer type`有`conv`、`logistics loss`、`conv transpose`和`sigmoid`。`conv`表示卷积网络，卷积层的参数有`kernel size`、`stride`、`pad`、`dilate`和`output channel`。`kerel size`为卷积核的大小，在这里的`kernel`的高和宽都是相等的，也可以设置成不等的。同理，`stride`、`dilate`都可以设置成宽高不等的，设置成不相等的时候需要使用一个向量来表示，如`[h,w]` 。`stride`表示步长，`stride>1`表示将卷积得到的结果进行下采样。`pad`表示网络层操作之前对图像数据的上下左右进行补零，`pad=0`表示没有对图像数据进行补零，`pad`也可以是一个向量，格式是`[top bottom left right] `。`dilate`表示扩张率，用于扩张卷积的使用，`dilate=1`时表示为普通卷积，`dilate>1`为扩张卷积。`output channel`表示输出的特征图的个数。

如代码1所示，是在网络中添加一个卷积层。`net=dagnn.DagNN()`
，表示新建一个空的网络，使用`net.addLayer()`向网络中添加网络层。`net.addLayer()`的使用为`net.addLayer(layer_name,layer,input_vars_name,output_vars_name,params_name)` 。

其中，`layer_name`为网络层的名字，网络层的名字需要在整个网络中是唯一的。`input_vars_name`为网络层的输入变量的名字，可以是一个`string`也可以是一个`cell`数组，这些输入变量的名字在该网络中的所有变量中也是唯一的，不具有二义性，让网络在运行的时候可以准确地找到该变量。

当`input_vars_name`是一个`string`时，表示网络层只有一个输入，当是一个长度大于1的`cell`数组时，表示当前网络层有多个输入。`output_vars_name`表示网络层的输出变量的名字，同理，可以有一个或多个输出变量。`params_name`表示网络层参数的名字，比如当前网络层是一个卷积层的时候，有`filters`和`biases`这两个参数。当网络层没有训练参数的时候，就不需要输入`params_name` 。

`net.addLayer()`的第二个参数是`layer`，表示一个网络层的实例。在代码1中，这个`layer`是一个卷积层的实例。使用`dagnn.Conv()`生成了一个卷积层的实例，输入的参数有`size`，`stride`、`pad`、`dilate`和`hasBias`，分别是卷积核的大小、步长、图像上下左右的`padding`、扩张率，以及卷积层中是否有`biases`参数。注意到这里的卷积核的`size`是一个`1×4`的向量，分别`[height width input_channel output_channel]`，当`hasBias`为`false`的时候，卷积层没有`biases`参数，此卷积层的参数就只有卷积核。

这里进行小结，添加一个网络层的时候，需要指定网络层的名字、网络层的实例、输入变量的名字、输出变量的名字和网络层参数名字，这些名字在网络中需要是唯一的。通过指定唯一的名字，网络在运行的时候会到对应的`struct array`中获取输入变量，或是将运行结果存放到对应的位置。

```matlab
% 代码1 添加一个卷积层
net = dagnn.DagNN();
net.addLayer('conv_layer_name',... % layer name
    dagnn.Conv('size', [4,4,3,1], 'stride', 1, 'pad', 0, 'dilate', 1, 'hasBias', true),... % layer
    'input_var_name',... % input var name
    'output_var_name',... % output var name
    {'filters_name', 'biases_name'}); % params name
```

如代码2所示，是向生成器中添加一个`logistics loss`层。在`net.addLayer()`函数的输入参数中，`loss_layer`是当前网络的层的名字，`dagnn.Loss()`是一个`loss`层的实例，`dagnn.Loss`中有多种`loss`供选择，在这里使用`logistics loss`，如要使用`softmaxlog`可以表示为`dagnn.Loss('loss','softmaxlog')` 。而`{'conv_5_output','labels'}`表示网络层的输入变量分别是`conv_5_output`和`labels` 。`net.addLayer()`的第四个参数`loss`，是输出变量的名字。由于该网络层没有参数，所以传入训练参数的名字。

```matlab
%代码2 添加一个logistics loss层
net.addLayer('loss_layer',...
    dagnn.Loss('loss', 'logistic'), ...
    {'conv_5_output', 'labels'},...
    'loss');
```
如代码3所示，表示向网络中添加一个反卷积层，其中的一些参数之前已经介绍过了，这里不再赘述。讲一些添加反卷积层和卷积层的不同之处，反卷积的卷积核为`[height width output_channel input_channel]`，注意到，`output_channel`和`input_channel`的位置与卷积层的使用方式相反，`upsample`表示上采样，`upsample=1`的时候为正常卷积，`upsample>1`的时候，先对输入数据进行上采样再进行卷积操作，可以将这里的`upsample`看成是步长。

```matlab
%代码3 添加一个conv transpose层
net.addLayer('convt_6',... % layer name
    dagnn.ConvTranspose('size', [4,4,1,16], 'upsample', 1, 'crop', 0, 'hasBias', true),... % layer
    last_added.var,... % input var name
    'convt_6_output',...% output var name
    {'convt_6_filters', 'convt_6_biases'}); % params name
```
如代码4所示，表示向网络中添加一个`sigmoid`激活层。
```matlab
% 代码4 添加一个sigmoid激活层
net.addLayer('sigmoid_layer', ...
    dagnn.Sigmoid(), ...
    'convt_6_output',...
    'generator_output');
```

按照上面所说的添加网络层的方式可以将整个网络结构搭建出来。其他的、没有出现在表1、2中的，鉴别器的前4层卷积操作后面都添加了`batch norm`层和`leaky rate=0.2`的`relu`层，生成器的前5层反卷积层后面都添加了`batch norm`层和`leaky rate=0.2`的`relu`层。

如代码5所示，向网络中添加一个`batch norm`层，如代码6所示，向网络中添加一个`relu`层，跟之前添加网络层的方法相同，需要指定网络层的名字、输入变量的名字、输出变量的名字和网络层参数的名字。

```matlab
%代码5 添加一个batch norm 层
net.addLayer('batch_norm_layer', ... % layer name
    dagnn.BatchNorm('numChannels', in_channels, 'epsilon', 1e-5), ... % layer
    'convt_output', ... % input var name
    'batch_norm_output', ... % output var naem
    {'bn_w', 'bn_b', 'bn_m'}) ; % params name
```
```matlab
%代码6 添加一个relu层
net.addLayer('relu_layer', ...
    dagnn.ReLU('leak', 0.2), ...
    'batch_norm_output',...
    'relu_output');
```
生成器和鉴别器都搭建好了之后，使用`initParams()`函数，对网络中的所有参数进行随机的初始化。除此之外，对于鉴别器来说，我们需要使用到`net.vars(net.getVarIndex('loss')).precious=1`，对鉴别器中的`loss`变量进行保留，`loss`变量是鉴别器`loss layer`的输出变量，通过`net.getVarIndex('loss')`获取到`loss`变量在鉴别器`vars`中的`index`，并且将这个`var`的`precious`的字段设置为1，表示这个变量是一个重要变量，在网络进行正向、反向传播之后，不需要将这个变量中的内容进行清空。同样地，需要使用到`net.vars(net.getVarIndex('generator_output')).precious=1`，对生成器中的`generator_output`变量进行保留。



### 4.  下载训练数据
训练所使用的数据集是`mnist`手写数字数据集。数据集在[这里](http://yann.lecun.com/exdb/mnist/)下载，网站不只提供了手写数字图像，还提供了图像对应的正确数字标签，但训练`GAN`网络只需要使用到手写数字图像。

如代码7所示，下载完毕之后，可以使用`matlab`中的`fopen`函数打开文件，并使用`fread`进行读取，读取出来的数组包含其他信息，图像数据从第17行开始。因为所有的图像都被拉成了一维向量并拼接在了一起，所以还需要将数据进行`reshape`，再使用`permute`交换图像的第一、二个维度。
```matlab
%代码7 读取图像数据
f=fopen(fullfile(dataDir, 'train-images-idx3-ubyte'),'r') ;
     images1=fread(f,inf,'uint8');
     fclose(f) ;
     images1=permute(reshape(images1(17:end),28,28,60e3),[2 1 3]) ;
```
###4.  训练网络
网络搭建好了，手写数字也下载好了，接下来就可以对生成器和鉴别器进行训练。在本文训练网络时，只是简单地将以`1：1`的比例对训练鉴别器和生成器进行迭代更新，每一轮迭代使用`batch size`个数据对网络进行训练。



### 5 训练网络

#### 5.1  对鉴别器进行更新
首先更新鉴别器，如代码8所示，先使用随机生成的噪声传入到`generator`中，`generator`使用`eval`函数进行了一次前向传播之后，得到`fake image`。再使用`fake image`和`fake label`对`discriminator`进行前向和反向传播得到`discriminator`网络参数的偏导数，到这里还没有结束，再使用`real image`和`real label`对`discriminator`进行前向和反向传播得到偏导数，将两次的偏导数相加对网络进行梯度下降。

在代码8中，网络的前向传播和反向传播都是使用`eval`函数，如果只需要进行前向传播，向`eval`中传入训练数据即可。如`generator.eval({'noises',batch_noise})` ，使用`eval`函数对`generator`中传入了随机的噪声`batch_noise`，`batch_noise`对应的`generator`中的变量为`noises`，`generator`就可以确定此次传播从以`noises`变量为输入的网络层开始。

进行反向传播的时候，也是使用`eval`函数，只不过需要传入两个参数，第一个参数是输入数据，第二个参数，是反向传播的偏导数，这两个参数都是`cell `数组。如`discriminator.eval({'images',batch_fake_images,'labels',batch_fake_labels},{'loss',1})`。`{'images',batch_fake_images,'labels',batch_fake_labels}`就是前向传播的输入，表示完成前向传播需要给定`images`和`labes`变量。而`{'loss',1}`表示反向传播从以`loss`变量为输出的网络层开始，1表示整个网络的损失函数对`loss`变量的偏导数，因为整个网络的损失函数就是`loss`，所以这里的偏导数就等于1。

在代码8中，还使用到了`discriminator.accumulateParamDers=0` 这句代码，将`accumlateParamDers`参数设置为`0`，表示反向传播时会对网络中的偏导数变量进行覆盖写入。将`accumlateParamDers`设置成`1`，表示进行反向传播得到的偏导数会与之前得到的对应的偏导数进行相加，而非覆盖写入。使用到了这两句代码，是因为在这里对`discriminato`r进行了两次反向传播，第一次传入网络中的数据是`fake image`和`fake label`，第二次传入网络中的数据是`real image`和`real label`。需要将两次反向传播得到的偏导数进行相加，再对网络进行更新。

对网络进行更新的时候可以使用到最简单的梯度下降的方式，也可使用`MatConvNet`提供的优化器对网络进行更新。损失函数对各个网络参数的偏导数保存在`net.params()`这个`struct array`的`der`字段中，损失函数对各个网络变量的偏导数保存在`net.vars()`这个`struct array`的`der`字段中。

代码8中的`stateD`是一个`cell array`，它是用于优化器更新网络的辅助变量，或是`momentum`类更新算法的辅助变量。

```matlab
%代码8 更新鉴别器
generator.eval({'noises', batch_noise});
batch_fake_images = generator.getVar('generator_output');
batch_fake_images = batch_fake_images.value;
 
discriminator.accumulateParamDers = 0;
discriminator.eval({'images', batch_fake_images, 'labels', batch_fake_labels}, {'loss', 1});
d_loss = discriminator.getVar('loss').value;
 
discriminator.accumulateParamDers = 1;
discriminator.eval({'images', batch_real_images, 'labels', batch_real_labels}, {'loss', 1});
d_loss = d_loss + discriminator.getVar('loss').value;

stateD = update_network(discriminator, stateD, params);
```
#### 5.2  对鉴别器进行更新

更新完鉴别器之后，就需要对生成器进行更新，如代码9所示，先让`generator`使用噪声进行前向传播，得到`fake image`。将`fake image`和`fake label`传入`discriminator`进行前向和反向传播，得到损失函数对`fake image`的偏导数，再将这个偏导数传入`generator`进行前向和反向传播，得到损失函数对`generator`各个网络参数的偏导数，以更新`generator`。从这里可以看出`generator`的`bp`的过程需要先经过`discriminator`的`bp`。

在代码9中，`generator`先进行前向传播之后，使用`getVar`函数得到`fake image`，在构造生成器的时候，生成器的最终输出变量的名称为`generator_output`，所以使用`getVar('generator_output')`得到对应的结构体，在`value`字段中存放的是`fake image`。而损失函数对`generator`的最终输出的偏导数，需要从`discriminator`中获取，该偏导数存放在`discriminator`的`images`变量中，使用`getVar`函数获取到`discriminator`的`images`变量，再从这个变量的`der`字段中获取到偏导数。

```matlab
% 代码9 更新生成器
generator.eval({'noises', batch_noise});
batch_fake_images = generator.getVar('generator_output');
batch_fake_images = batch_fake_images.value;
 
discriminator.accumulateParamDers = 0;
discriminator.eval({'images', batch_fake_images, 'labels', batch_real_labels}, {'loss', 1});
g_loss = discriminator.getVar('loss').value;
der_from_discriminator = discriminator.getVar('images');
der_from_discriminator = der_from_discriminator.der;
 
generator.accumulateParamDers = 0;
generator.eval({'noises', batch_noise}, {'generator_output', der_from_discriminator});
stateG = update_network(generator, stateG, params);
```
#### 5.3  更新网络的方式
如代码10所示，是使用梯度下降的方式对网络的参数进行更新，这是最简单的参数更新方式。遍历整个网络中的参数，从`net.params(p).der`获取到参数对应的偏导数，进行梯度下降。其中的`vl_taccum(alpha,a,beta,b)`函数，实现的功能就是`alpha×a+beta×b`。
```matlab
%代码10 梯度下降更新网络
for p=1:numel(net.params)
    parDer = net.params(p).der ;
    net.params(p).value = vl_taccum(...
        1,  net.params(p).value, -thisLR, parDer) ;
end
```
如代码11所示，是使用优化器来更新网络的例子，其中的`solver()`对应某个`MatConvNet`提供的优化器函数。`state`是辅助变量，该辅助变量是一个`cell`数组，`cell`中的单元个数与网络的`params`数组长度一致，初始化的时候，`cell`数组中的所有单元均为0。
```matlab
% 代码11 使用优化器来更新网络
for p=1:numel(net.params)
    parDer = net.params(p).der ;
    [net.params(p).value, state.solverState{p}] = ...
        solver(net.params(p).value, state.solverState{p}, ...
        parDer, solverOpts, thisLR) ;
end
```
使用上面提及的网络更新方式，对生成器和鉴别器进行迭代地更新，固定一个更新一个，以完成`GAN`网络的训练。



### 6 保存训练好的网络模型
如代码12所示，是保存网络模型的代码，使用`saveobj`函数，将`generator`和`discriminator`转化成一个结构体，并使用`save`函数将结构体保存在`.mat`文件中。
```matlab
% 代码12 保存网络模型
function save_model_fun(path, generator_, discriminator_, stateG, stateD)
    generator = generator_.saveobj();
    discriminator = discriminator_.saveobj();
    save(path, 'generator', 'discriminator', 'stateG', 'stateD');
end
```
如代码13所示，是加载网络模型的代码，先使用`load`函数从`.mat`文件中读取网络模型的结构体，再使用`loadobj`将结构体转化为`dagnn.DagNN`类实例。
```matlab
% 代码13 加载网络模型
function [generator, discriminator, stateG, stateD] = load_model_fun(path)
    load(path, 'generator', 'discriminator', 'stateG', 'stateD');
    generator = dagnn.DagNN.loadobj(generator);
    discriminator = dagnn.DagNN.loadobj(discriminator);
end
```


### 7 使用gpu进行加速

使用`gpu`来加速网络的训练，需要将`MatConvNet`的代码编译成`gpu`版本。机器上要安装好`Nvidia`的`gpu`和`cuda`，而`cudnn`则可以根据需要安装。

编译`MatConvNet`的代码时，使用的是`vl_compilenn`函数，如命令1所示，是将`MatConvnet`编译成`gpu`版本的例子，其中指定了`cuda`和`cudnn`的安装路径。注意，编译的时候，是在`matlab`的命令行窗口进行输入命令的，编译将`MatConvNet`中的`C/C++`代码编译成`mex`格式。在`windows`平台上进行编译时需要安装`visual studio`，在`linux`平台上需要安装`gcc`、`g++`。
```matlab
%命令1 编译代码的例子
vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
               'cudaRoot', '/Developer/NVIDIA/CUDA-6.5', ...
               'enableCudnn', true, 'cudnnRoot', 'local/cudnn-rc2') ;
```
####6.1  使用单个gpu进行加速
使用单个`gpu`进行加速时，需要先将`MatConvNet`编译成`gpu`版本，接下来使用`gpuDevice()`函数来获取`gpu`设备。获取到`gpu`设备之后，将传入网络中的数据转换成`gpu array`，并且使用`move`函数，将`generator`和`discriminator`移动到`gpu`上，如代码14所示。其他的部分与没有使用`gpu`进行加速的代码一致。
```matlab
%代码14 将网络移动到gpu上
generator.move('gpu') ;
discriminator.move('gpu');
```
####6.2 使用多个gpu进行加速
使用多个`gpu`进行加速的时候，同样地，需要将代码编译成`gpu`版本、将训练数据转换成`gpu array`、将网络移动到`gpu`上。除此之外，还需要使用到`gcp`函数来创建一个并行池，并行池`worker`的数目与需要使用到的`gpu`数目一致。如代码15所示，使用`spmd`让并行池中的每一个`worker`都申请一个`gpu`设备。
```matlab
%代码15 申请多个gpu
spmd
    gpuDevice(gpus(labindex))
end
```
如代码16所示，为了让训练部分并行，需要为`generator`和`discriminator`各自设置一个`ParameterServer`实例，设置之后，`generator`和`discriminator`的网络参数的偏导数都需用从该`ParameterServer`中获取，而`generator`和`discriminator`自身不再保存网络参数的偏导数。

训练部分的代码同样需要使用`spmd`语句进行并行，将`batch size`的训练数据划分给多个`worker`进行网络训练。这里进行划分的方式很简单，如代码16所示，通过确定`batch_index_start`这个变量来进行划分，`labindex`是并行时每一个`worker`的`index`，`numlabels`是并行的`worker`的个数。

```matlab
%代码16 并行训练网络
generator.move('gpu') ;
discriminator.move('gpu');
if numGpus > 1
    parameterServer.method = 'mmap' ;
    parameterServer.prefix = 'mcn' ;
    
    parservG = ParameterServer(parameterServer) ;
    generator.setParameterServer(parservG);
    
    parservD = ParameterServer(parameterServer) ;
    discriminator.setParameterServer(parservD);
else
    parservG = [] ;
    parservD = [];
end
 
spmd
    for j=1:batch_count
        if j < batch_count
            batch_index_start = (j-1)* params.batch_size + 1 + (labindex-1);
            batch_index_end = j* params.batch_size;
        else
            batch_index_start = (j-1)* params.batch_size + 1 + (labindex-1);
            batch_index_end = size(real_images,4);
        end
        % train network here
        batch_real_images = real_images(:,:,:,batch_index_start : numlabs : batch_index_end);
    end
end
```
如代码17所示，更新网络的时候，网络参数的偏导数需要从对应的`ParameterServer`中获取，使用`pullWithIndex`函数来获取。
```matlab
%代码17 从ParameterServer中获取偏导数
for p=1:numel(net.params)
    parDer = parserv.pullWithIndex(p);
    net.params(p).value = vl_taccum(...
        1,  net.params(p).value, -thisLR, parDer) ;
end
```
其他需要注意的点是，如代码18所示，在从鉴别网络进行反向传播的时候，在`eval`函数中使用到了`holdOn`参数。`holdOn`参数为`true`，表示网络此次进行反向传播得到的偏导数不同步到`ParameterServer`中。当`holdOn`为`false`的时候，反向传播得到的偏导数就会`pull`到`ParameterServer`中，并将网络自身的偏导数清空。

`holdOn`参数默认为`false`，如果在`discriminator`使用`fake image`进行反向传播时候`holdOn`为`false`，训练时的网络参数的偏导数就会被`pull`到`ParameterServer`中，并且网络本身的偏导数会被清空，从而无法达成两次偏导数相加的要求，所以进行第一次反向传播的时候，`holdOn`要为`true`。
```matlab
%代码18 holdOn参数的使用
generator.eval({'noises', batch_noise});
batch_fake_images = generator.getVar('generator_output');
batch_fake_images = batch_fake_images.value;
 
% set the accumulateParamDers to 0, the derivative of the
% params with repects to loss will be overwrite
discriminator.accumulateParamDers = 0;
discriminator.eval({'images', batch_fake_images, 'labels', batch_fake_labels}, {'loss', 1}, 'holdOn', 1);
d_loss = discriminator.getVar('loss').value;
% set the accumlateParamsDers to 1, the derivative is equal to
% the old one plus the new one
discriminator.accumulateParamDers = 1;
discriminator.eval({'images', batch_real_images, 'labels', batch_real_labels}, {'loss', 1}, 'holdOn', 0);
d_loss = d_loss + discriminator.getVar('loss').value;
```
其实，进行两次反向传播可以替换成另外一种形式：将`fake image`和`real image`拼接在一起之后传入`discriminator`进行反向传播。
### 8 参考文献
1. `http://www.vlfeat.org/matconvnet/`
2. `https://github.com/vlfeat/matconvnet`