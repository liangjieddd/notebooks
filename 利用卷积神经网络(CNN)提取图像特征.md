[TOC]

# 利用卷积神经网络(CNN)提取图像特征

## 一、前言

​		本篇文章主要介绍了CNN网络中卷积层的计算过程，欲详细了解CNN的其它信息可以参考：技术向：一文读懂卷积神经网络。

​		卷积神经网络(CNN)是局部连接网络。相对于全连接网络其最大的特点就是：**局部连接性和权值共享性**。

​		因为对一副图像中的某个像素p来说，一般离像素p越近的像素对其影响也就越大（局部连接性）；另外，根据自然图像的统计特性，某个区域的权值也可以用于另一个区域(权值共享性)。

​		这里的权值共享说白了就是**卷积核共享**，对于一个卷积核将其与给定的图像做卷积就可以提取一种图像的特征，不同的卷积核可以提取不同的图像特征。概况的讲，卷积层的计算方法就是根据公式
$$
conv=\sigma (imgMat\circ W+b)  (1)
$$

- "σ"表示激活函数;
- "imgMat"表示灰度图像矩阵; 
- "W"表示卷积核;
- "∘ "表示卷积操作；
- "b "表示偏置值。

## 二、举例说明

​		下面用一个具体例子来详细说明卷积层的计算过程。

​		用到的图像为lena图像，如图1所示；

​		卷积核为Sobel卷积核，如图2所示。

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517221534494)

​														图1 Lena图像(512x512)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517221624107)

​							图2 Sobel卷积核(Gx表示水平方向，Gy表示垂直方向)

1、首先用Sobel—Gx卷积核来对图像做卷积，即公式(1)中的
$$
imgMat\circ W
$$
​		这里卷积核大小为3x3，图像大小为512x512如果不对图像做任何其它处理，直接进行卷积的话，卷积后的图像大小应该是:(512-3+1)x(512-3+1)。对卷积不懂的可以参考技术向：一文读懂卷积神经网络或其他读物。最终结果为：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517221849011)

​							图3 lena图像与Sobel—Gx卷积核的卷积结果

2、 将步骤1中所得结果(一个矩阵)的每个元素都加上b(偏置值)，并将所得结果(矩阵)中的每个元素都输入到激活函数，这里取sigmoid函数如下式所示
$$
f(x)=\frac { 1 }{ 1+{ e }^{ -x } } (2)
$$
最终结果如图4所示：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517222055981)

​													图4 卷积层所得到的最终结果

3、同理

​		利用Sobel—Gy卷积核我们最终可以得到如图5所示的结果。

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517222156818)

​									图5 Sobel—Gy卷积核卷积层所得到的最终结果

## 三、代码

~~~
clear
clc
imgRGB = imread('lena.jpg');
imgGray = double(rgb2gray(imgRGB));

Gx = [-1 0 1;-2 0 2;-1 0 1];
convImg = conv2(imgGray,Gx,'valid');
whos convImg
figure
subplot(1,2,1);
imshow(uint8(convImg));
title('Sobel-Gx卷积结果')
b = 0.2;
sigmImg = 1./(1+exp(-convImg)) + b;
subplot(1,2,2);
imshow(sigmImg);
title('Sobel-Gx-sigmoid函数激活结果')

Gy = [-1 0 1;-2 0 2;-1 0 1]';
convImg = conv2(imgGray,Gy,'valid');
whos convImg
figure
subplot(1,2,1);
imshow(uint8(convImg));
title('Sobel-Gy卷积结果')
b = 0.2;
sigmImg = 1./(1+exp(-convImg)) + b;
subplot(1,2,2);
imshow(sigmImg);
title('Sobel-Gy-sigmoid函数激活结果')
~~~

结果：

1、Sobel—Gx卷积核结果

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517222659545)

2、Sobel—Gy卷积核结果

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20160517222729139)

