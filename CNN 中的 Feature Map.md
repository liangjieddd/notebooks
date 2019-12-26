[TOC]

# CNN 中的 Feature Map

## 1.feature map的含义

​      在每个卷积层，数据都是以三维形式存在的。你可以把它看成许多个二维图片叠在一起，其中每一个称为一个feature map。

​		在输入层，如果是灰度图片，那就只有一个feature map；如果是彩色图片，一般就是3个feature map（红绿蓝）。

​		层与层之间会有若干个卷积核（kernel），上一层和每个feature map跟每个卷积核做卷积，都会产生下一层的一个feature map。

 

​		feature map（下图红线标出） 即：**该层卷积核的个数**，**有多少个卷积核，经过卷积就会产生多少个feature map**，也就是下图中 `豆腐皮儿`的层数、同时也是下图`豆腐块`的深度（宽度）！！

​		这个宽度可以手动指定，一般网络越深的地方这个值越大，因为随着网络的加深，feature map的长宽尺寸缩小，本卷积层的每个map提取的特征越具有代表性（精华部分），所以后一层卷积层需要增加feature map的数量，才能更充分的提取出前一层的特征，一般是成倍增加（不过具体论文会根据实验情况具体设置）！

![img](https://img-blog.csdn.net/20180605134623321?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxNTQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 

![img](https://img-blog.csdn.net/20180605134717740?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxNTQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 

 

![img](https://img-blog.csdn.net/20180605135241270?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MjMxNTQ5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

​       

​		卷积特征的可视化，有助于我们更好地理解深度网络。**卷积网络在学习过程中保持了图像的空间结构，**也就是说最后一层的激活值（feature map）总和原始图像具有空间上的对应关系，具体对应的位置以及大小，可以用感受野来度量。利用这点性质可以做很多事情：

1 前向计算。

​		我们直接可视化网络每层的 feature map，然后观察feature map 的数值变化. 一个训练成功的CNN 网络，**其feature map 的值伴随网络深度的增加，会越来越稀疏。**这可以理解网络取精去燥。

2 反向计算。

​		根据网络最后一层最强的激活值，利用感受野求出原始输入图像的区域。可以观察输入图像的那些区域激活了网络，**利用这个思路可以做一些物体定位。**

## 2. 结构

​		下图显示了CNN中最重要的部分，这部分称之为过滤器(filter)或内核(kernel)。

​		因为TensorFlow官方文档中将这个结构称之为过滤器(filter)，故在本文中将统称这个结构为过滤器。

​		如下图1所示，过滤器可以将当前层网络上的一个子节点矩阵转化为下一层神经网络上的一个单位节点矩阵。单位节点矩阵指的是长和宽都是1，但深度不限的节点矩阵。

![img](https://img-blog.csdn.net/20171208235343379?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQWxsZW5semNvZGVy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

​										图1：卷积层过滤器(filter)结构示意图

​       在一个卷积层中，过滤器所处理的节点矩阵的长和宽都是由人工指定，此节点矩阵的尺寸也称之为过滤器的尺寸。常用的过滤器尺寸有3\*3或5*5。

​		因为过滤器处理的矩阵深度和当前层神经网络的深度是一致的，所以虽然节点矩阵是三维的，但是过滤器的尺寸只需要指定两个维度。

​       过滤器中另外一个需要人工指定的设置处理得到的单位节点矩阵的深度，此设置称为过滤器的深度(此深度=过滤器的个数=feature map的个数)。

​       注意过滤器的尺寸指的是一个过滤器输入节点矩阵的大小，而深度指的是输出单位节点矩阵的深度。如图1中，左侧小矩阵的尺寸为过滤器的尺寸，而右侧单位矩阵的深度为过滤器的深度。

​       这里参考一篇文章A gentle dive into the anatomy of a Convolution layer.
​       链接：https://pan.baidu.com/s/1pLoQwGz

​       此文阐述了卷积层的一个特定的解剖特征。许多卷积架构是从一个外部卷积单元开始的，它将信道RGB的输入图像映射到一系列内部过滤器中。在当下最通用的深度学习框架中，这个代码可能如下所示：

~~~
out_1=Conv2d(input=image, filter=32, kernel_size=(3,3), strides=(1,1));//卷积层

relu_out=relu(out_1); //利用激活函数ReLU去线性化

//最大池化降维
pool_out=MaxPool(relu_out, kernel_size=(2,2),strides=2);

~~~

​		很容易理解，上面的结果是一系列的具有32层深度的过滤器。我们不知道的是，该如何将具有3个信道的图像精确地映射到这32层中！另外，我们也不清楚该如何应用最大池(max-pool)操作符。例如，是否一次性将最大池化应用到了所有的过滤层中以有效地生成一个单一的过滤映射？又或者，是否将最大池独立应用于每个过滤器中，以产生相同的32层的池化过滤器？
​       具体如何做的呢？
​       一图胜千言，下图可以显示上述代码片段中所有的操作。

![img](https://img-blog.csdn.net/20171209171829551?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQWxsZW5semNvZGVy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

​																				图2：卷积层的应用

​       观察上图，可以看到最显著的一点是，步骤1中的每个过滤器(即Filter-1、Filter-2……)实际上包含一组3个权重矩阵(Wt-R、Wt-G和WT-B)。每个过滤器中的3个权重矩阵分别用于处理输入图像中的红(R)、绿(G)和蓝(B)信道。

​		在正向传播期间，图像中的R、G和B像素值分别与Wt-R、Wt-G和Wt-B权重矩阵相乘以产生一个间歇激活映射(intermittent activation map)(图中未标出)，然后将这3个权重矩阵的输出(即3个间歇激活映射)相加以为每个过滤器产生一个激活映射(activation map)。

​       随后，这些激活映射中的每一个都要经过激活函数ReLu去线性化，最后到最大池化层，而后者主要为激活映射降维。

​		最后，我们得到的是一组经过激活函数和池化层处理后的激活映射，现在其信号分布在一组32个(过滤器的数量)二维张量之中(也具有32个feature map，每个过滤器会得到一个feature map)。

​       来自卷积层的输出经常用作后续卷积层的输入。因此，如果我们的第二个卷积单元如下：

​       conv_out_2 = Conv2d（input = relu_out，filters = 64）

​       那么框架就需要实例化64个过滤器，每个过滤器使用一组32个权重矩阵。



## 3.计算

​		一直以来，感觉 feature map 挺晦涩难懂的，今天把初步的一些理解记录下来。参考了斯坦福大学的机器学习公开课和七月算法中的机器学习课。

​        CNN一个牛逼的地方就在于**通过感受野和权值共享减少了神经网络需要训练的参数的个数**。总之，卷积网络的核心思想是将：局部感受野、权值共享（或者权值复制）以及时间或空间亚采样这三种结构思想结合起来获得了某种程度的位移、尺度、形变不变性.

​       下图左：如果我们有1000x1000像素的图像，有1百万个隐层神经元，那么他们全连接的话（每个隐层神经元都连接图像的每一个像素点），就有1000x1000x1000000=10^12 个连接，也就是10^12 个权值参数。

​		然而图像的空间联系是局部的，就像人是通过一个局部的感受野去感受外界图像一样，**每一个神经元都不需要对全局图像做感受，**每个神经元只感受局部的图像区域，然后在更高层，将这些感受不同局部的神经元综合起来就可以得到全局的信息了。

​		这样，我们就可以减少连接的数目，也就是减少神经网络需要训练的权值参数的个数了。如下图右：假如局部感受野是10x10，隐层每个感受野只需要和这10x10的局部图像相连接，所以1百万个隐层神经元就只有一亿个连接，即10^8个参数。比原来减少了四个0（数量级），这样训练起来就没那么费力了，但还是感觉很多的啊，那还有啥办法没？

![img](https://img-blog.csdn.net/20131230210240250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbmFuMzU1NjU1NjAw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

​       我们知道，隐含层的每一个神经元都连接10x10个图像区域，也就是说每一个神经元存在10x10=100个连接权值参数。那如果我们每个神经元这100个参数是相同的呢？

​		也就是说每个神经元用的是同一个卷积核去卷积图像。这样我们就只有多少个参数？？只有100个参数啊！不管你隐层的神经元个数有多少，两层间的连接我只有100个参数啊！这就是权值共享。

​		好了，你就会想，这样提取特征也忒不靠谱吧，这样你只提取了一种特征啊？对了，真聪明，我们需要提取多种特征对不？

​		假如一种滤波器，也就是一种卷积核就是提出图像的一种特征，例如某个方向的边缘。那么我们需要提取不同的特征，怎么办，加多几种滤波器不就行了吗？

​		所以假设我们加到100种滤波器，每种滤波器的参数不一样，表示它提出输入图像的不同特征，例如不同的边缘。

​		这样每种滤波器去卷积图像就得到对图像的不同特征的放映，我们称之为Feature Map。所以100种卷积核就有100个Feature Map。这100个Feature Map就组成了一层神经元。

​		到这个时候明了了吧。我们这一层有多少个参数了？100种卷积核x每种卷积核共享100个参数=100x100=10K，也就是1万个参数。才1万个参数。见下图右：不同的颜色表达不同的滤波器。

  ![img](https://img-blog.csdn.net/20131230210323484?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbmFuMzU1NjU1NjAw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



​       嘿哟，遗漏一个问题了。刚才说隐层的参数个数和隐层的神经元个数无关，只和滤波器的大小和滤波器种类的多少有关。

​		那么隐层的神经元个数怎么确定呢？它和原图像，也就是输入的大小（神经元个数）、滤波器的大小和滤波器在图像中的滑动步长都有关！

​		例如，我的图像是1000x1000像素，而滤波器大小是10x10，假设滤波器没有重叠，也就是步长为10，这样隐层的神经元个数就是(1000x1000 )/ (10x10)=100x100个神经元了，假设步长是8，也就是卷积核会重叠两个像素，那么……我就不算了，思想懂了就好。注意了，这只是一种滤波器，也就是一个Feature Map的神经元个数哦，如果100个Feature Map就是100倍了。由此可见，图像越大，神经元个数和需要训练的权值参数个数的贫富差距就越大。

**feature map计算方法：**

在CNN网络中roi从原图映射到feature map中的计算方法:

​		INPUT为32\*32，filter的大小即kernel size为5\*5，stride = 1，pading=0,卷积后得到的feature maps边长的计算公式是： 
output_h =（originalSize_h+padding*2-kernelSize_h）/stride +1 

​		所以，卷积层的feature map的边长为：

​		conv1_h=（32-5）/1 + 1 = 28 

​		卷积层的feature maps尺寸为28\*28. 

​		由于同一feature map共享权值，所以总共有6\*（5\*5+1）=156个参数。 



​		卷积层之后是pooling层，也叫下采样层或子采样层（subsampling）。它是利用图像局部相关性的原理，对图像进行子抽样，这样在保留有用信息的同时可以减少数据处理量。

​		pooling层不会减少feature maps的数量，只会缩减其尺寸。常用的pooling方法有两种，一种是取最大值，一种是取平均值。 

​		pooling过程是非重叠的，S2中的每个点对应C1中2\*2的区域（也叫感受野），也就是说

kernelSize=2，stride=2，所以:

pool1_h = (conv1_h - kernelSize_h)/stride +1 = (28-2)/2+1=14。

pooling后的feature map尺寸为14*14.



​		fast rcnn以及faster rcnn做检测任务的时候，涉及到从图像的roi区域到feature map中roi的映射，然后再进行roi_pooling之类的操作。  

​       比如图像的大小是（600,800），在经过一系列的卷积以及pooling操作之后在某一个层中得到的feature map大小是（38,50），那么在原图中roi是（30,40,200,400），

在feature map中对应的roi区域应该是

roi_start_w = round(30 * spatial_scale);
roi_start_h = round(40 * spatial_scale);
roi_end_w = round(200 * spatial_scale);
roi_end_h = round(400 * spatial_scale);

其中spatial_scale的计算方式是

spatial_scale=round(38/600)=round(50/800)=0.0625，

所以在feature map中的roi区域[roi_start_w,roi_start_h,roi_end_w,roi_end_h]=[2,3,13,25];

具体的代码可以参见caffe中roi_pooling_layer.cpp