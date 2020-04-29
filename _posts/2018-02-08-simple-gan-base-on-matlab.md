---
layout: article
title:  "简单GAN网络matlab实现"
date:   2018-02-08 20:44:51 +0800
categories: [blog]
tags: [matlab, gan]
mathjax: false
---

### 1 概述
此代码在`matlab`上搭建了简单的生成对抗性网络，用来生成手写数字图像。
网络中生成器和鉴别器的隐藏层均为2层，且都是全连接层，是一个比较简单的网络结构。主要用来说明怎么在`matlab`上搭建`GAN（Generative Adversarial Net）`网络。

#### 1.1 网络模型
如图1所示，是生成器网络模型，一个输入层，两个全连接层。输入的数据是`100×1`的噪声，输出是`784×1`的向量，将输出进行`reshape`之后，就可以得到一张`28×28`的手写数字图像。
![图1 生成器](/assets/simple-gan-base-on-matlab/generator.png)
如图2所示，是将生成器和鉴别器连接起来之后的模型。这里主要想说明一点，在进行网络参数更新的时候，为了得到生成器参数的偏导数，`bp`过程需要鉴别器，再传到生成器。

到这里就产生了一个疑问：不是说更新生成器的时候，鉴别网络的参数需要固定住不变吗，如果bp过程需要经过鉴别网络，那应该怎么保持鉴别网络的参数不变呢？
其实`bp`的时候，只是算出来了各个网络层参数对于`loss`的偏导数，在求生成器的参数的偏导数的时候，鉴别网络的参数的偏导数也被求出来了。但是求出来了偏导数，不一定就要对网络进行更新。也就是求出来了网络`loss`对生成器和鉴别器的偏导数，但是只使用到了生成器的偏导数来更新生成器。
![图2 将generator和discriminator看成一个整体](/assets/simple-gan-base-on-matlab/generator-and-discriminator.png)


#### 1.2 More
* 这个是我写的、在`matlab`上实现`GAN`网络的另外一份代码，里面的网络模型使用到了卷积、反卷积等。
  
  [GAN 网络matlab实现]({% post_url 2018-06-01-gan-base-on-matlab %})
  
* 在`matlab`平台上，使用`MatConvNet`搭建`GAN`网络的例子

  [使用MatConvNet搭建GAN网络]({% post_url 2018-08-27-gan-base-on-matconvnet %})

* mnist_uint8.mat [下载地址](https://github.com/rasmusbergpalm/DeepLearnToolbox/tree/master/data)

### 2 实例
#### 2.1 实例1
  😛😜😝代码在`github`：[gan_adam.m](https://github.com/jonzhaocn/Simple-GAN-Base-on-Matlab/blob/master/gan_adam.m)
  这里使用上面提到的网络结构来生成手写数字图片，使用到了`Adam`算法作为优化器来更新`GAN`网络。

```matlab
clear;
clc;
% -----------加载数据
load('mnist_uint8', 'train_x');
train_x = double(reshape(train_x, 60000, 28, 28))/255;
train_x = permute(train_x,[1,3,2]);
train_x = reshape(train_x, 60000, 784);
% -----------------定义模型
generator = nnsetup([100, 512, 784]);
discriminator = nnsetup([784, 512, 1]);
% -----------开始训练
batch_size = 60;
epoch = 100;
images_num = 60000;
batch_num = ceil(images_num / batch_size);
learning_rate = 0.001;
for e=1:epoch
    kk = randperm(images_num);
    for t=1:batch_num
        % 准备数据
        images_real = train_x(kk((t - 1) * batch_size + 1:t * batch_size), :, :);
        noise = unifrnd(-1, 1, batch_size, 100);
        % 开始训练
        % -----------更新generator，固定discriminator
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        discriminator = nnff(discriminator, images_fake);
        logits_fake = discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));
        generator = nnbp_g(generator, discriminator);
        generator = nnapplygrade(generator, learning_rate);
        % -----------更新discriminator，固定generator
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        images = [images_fake;images_real];
        discriminator = nnff(discriminator, images);
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(batch_size,1);ones(batch_size,1)];
        discriminator = nnbp_d(discriminator, logits, labels);
        discriminator = nnapplygrade(discriminator, learning_rate);
        % ----------------输出loss
        if t == batch_num
            c_loss = sigmoid_cross_entropy(logits(1:batch_size), ones(batch_size, 1));
            d_loss = sigmoid_cross_entropy(logits, labels);
            fprintf('c_loss:"%f",d_loss:"%f"\n',c_loss, d_loss);
        end
        if t == batch_num
            path = ['./pics/epoch_',int2str(e),'_t_',int2str(t),'.png'];
            save_images(images_fake, [4, 4], path);
            fprintf('save_sample:%s\n', path);
        end
    end
end
% sigmoid激活函数
function output = sigmoid(x)
    output =1./(1+exp(-x));
end
% relu
function output = relu(x)
    output = max(x, 0);
end
% relu对x的导数
function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
% 交叉熵损失函数，此处的logits是未经过sigmoid激活的
% https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end
% sigmoid_cross_entropy对logits的导数，此处的logits是未经过sigmoid激活的
function result = delta_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
% 根据所给的结构建立网络
function nn = nnsetup(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v是应用adam算法更新网络所需的变量
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % 假设结构为[100, 512, 784]，则有3层，输入层100，两个隐藏层：100*512，512*784, 输出为最后一层的a值（激活值）
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end
% 前向传递
function nn = nnff(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.layers_count
        input = nn.layers{i-1}.a;
        w = nn.layers{i}.w;
        b = nn.layers{i}.b;
        nn.layers{i}.z = input*w + repmat(b, size(input, 1), 1);
        if i ~= nn.layers_count
            nn.layers{i}.a = relu(nn.layers{i}.z);
        else
            nn.layers{i}.a = sigmoid(nn.layers{i}.z);
        end
    end
end
% discriminator的bp，下面的bp涉及到对各个参数的求导
% 如果更改网络结构（激活函数等）则涉及到bp的更改，更改weights，biases的个数则不需要更改bp
% 为了更新w,b，就是要求最终的loss对w，b的偏导数，残差就是在求w，b偏导数的中间计算过程的结果
function nn = nnbp_d(nn, y_h, y)
    % d表示残差，残差就是最终的loss对各层未激活值（z）的偏导，偏导数的计算需要采用链式求导法则-自己手动推出来
    n = nn.layers_count;
    % 最后一层的残差
    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        nn.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% generator的bp
function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator的loss是由label_fake得到的，(images_fake过discriminator得到label_fake)
    % 对g进行bp的时候，可以将g和d看成是一个整体
    % g最后一层的残差等于d第2层的残差乘上(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
% 应用梯度
% 使用adam算法更新变量，可以参考：
% https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
function nn = nnapplygrade(nn, learning_rate)
    n = nn.layers_count;
    nn.t = nn.t+1;
    beta1 = nn.beta1;
    beta2 = nn.beta2;
    lr = learning_rate * sqrt(1-nn.beta2^nn.t) / (1-nn.beta1^nn.t);
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        % 下面的6行代码是使用adam更新weights与biases
        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w - lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db.*db);
        nn.layers{i}.b = nn.layers{i}.b - lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon); 
    end
end
% 保存图片，便于观察generator生成的images_fake
function save_images(images, count, path)
    n = size(images, 1);
    row = count(1);
    col = count(2);
    I = zeros(row*28, col*28);
    for i = 1:row
        for j = 1:col
            r_s = (i-1)*28+1;
            c_s = (j-1)*28+1;
            index = (i-1)*col + j;
            pic = reshape(images(index, :), 28, 28);
            I(r_s:r_s+27, c_s:c_s+27) = pic;
        end
    end
    imwrite(I, path);
end
```
结果
![epoch_5_t_1000.png](/assets/simple-gan-base-on-matlab/example1-result1.png)
![epoch_13_t_1000.png](/assets/simple-gan-base-on-matlab/example1-result2.png)

#### 2.2 实例2，Mini-batch Gradient Descent
😛😜😝代码在`github`：[gan_mbgd.m](https://github.com/jonzhaocn/Simple-GAN-Base-on-Matlab/blob/master/gan_mbgd.m)

这里使用的网络结构与实例1的一样，只是换了一种优化器。使用Mini-batch Gradient Descent算法更新GAN网络，下面是部分代码。
```matlab
% 应用梯度
function nn = nnapplygrade(nn, learning_rate)
    n = nn.layers_count;
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        nn.layers{i}.w = nn.layers{i}.w - learning_rate * dw;
        nn.layers{i}.b = nn.layers{i}.b - learning_rate * db;
    end
end
```
结果：
![epoch_11_t_1000.png](/assets/simple-gan-base-on-matlab/example2-result1.png)

![epoch_13_t_1000.png](/assets/simple-gan-base-on-matlab/example2-result2.png)

