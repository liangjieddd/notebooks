[TOC]

# DenseNet 算法详解（目标检测backbone）

## 1.算法背景

### 1.1 简介

论文名称：Densely Connected Convolutional Networks
论文链接：ttps://arxiv.org/abs/1608.06993
论文日期：2018.01.28
代码：https://github.com/liuzhuang13/DenseNet

​		若在**输入层与输出层之间包含更短的连接**，那么更深的神经网络结构会训练地更准确、更有效。因此在本文中，设计了DenseNet神经网络结构，其中的每一层都以**前馈**方式连接到其他每一层。
​		之前的神经网络有L层，就有L个连接，DenseNet网络含有**L(L+1)/2**个连接。对于每一层，所有前面的预测层的feature map都作为输入。然后输出的feature map作为后面所有层的输入。

### 1.2 优势

- 解决了梯度消失（vanishing-gradient）问题；
- 增强了特征传播；
- 增强了特征复用；
- 大大地减少了参数量。

​	     本文在CIFAR-10， CIFAR-100， SVHN与 ImageNet等数据集上进行了对比实验，DenseNets的准确率有了很大的提升。

## 2.算法简介

### 2.1 算法提出背景

​		由于计算机硬件的进步，允许神经网络结构变得更深。原始的LeNet5由5层组成，VGG由19层组成，去年的Highway Networks与ResNet超过了100层。
​		随着CNN神经网络结构变得更深，一个新的问题被提出：由于输入或梯度的信息通过了许多层，当到达神经网络的输出时可能会造成**梯度消失。**

### 2.2 其他算法解决方案

​		ResNets和Highway Networks通过**恒等连接**从一层绕过信号到下一层，随机深度通过在训练过程中**随机丢弃层**来缩短ResNet，以便提供更好的信息和梯度流。
​		FractalNets重复将**几个平行层序列**与不同数目的**卷积块**组合，得到一个大的**标称深度**，同时保持网络中的多条短路径。
​		这些方法都是为了创造了从靠近输入层的网络层到靠近输出层的网络层的短路径。

### 2.3 DenseNets算法

​		本文提出了DenseNets，为了确保网络层之间**最多的信息流**，本文将**所有的网络层与其它前面的层相连**，为了保留前馈特性，**每个层从前面的所有层获得额外的输入**，**并将自己的特征映射传递到后续的所有层**。

​		与ResNets相反，在特征传到网络层之前，从不通过求和来组合它们；而是通过**将它们连接起来组合它们**。

### 2.4 两大优势

		1. DenseNet一个最大的优势就是比传统卷积神经网络需要**更少的参数**，由于它不需要重复学习多余的feature-maps。
			传统的前馈结构是从一层传递到另一层，每一层从前一层神经网络读到状态，并且写入下一层。会改变状态，但是仍然传递需要保存的信息。ResNets通过**加性恒等变换**使这种信息保存显式化。

​		ResNets最近的变化表明，许多层的贡献很小，实际上可以在训练中随机丢弃这些层。但是ResNets的参数量是非常大的，因为每一层都有自己的权重。
	​		DenseNet体系结构明确区分了**添加到网络的信息和保留的信息**。
	​		DenseNet层**非常窄**，每层只有12个filters， 只向网络的“集体知识”添加**一小部分特征映射**，并保持其余特征图不变，最后的分类器根据网络中的所有特征映射进行决策。
	
  2. 除了更好的参数有效性，另一个优势是它们**改善了整个网络的信息流和梯度**， 会使训练更简单，每个网络层都可以直接从损耗函数和原始输入信号中访问梯度，从而进行隐式深度监督。这有助于对更深层次的网络体系结构进行训练。

     ​		密集的连接具有**正则化**效果，从而减少了对训练集规模较小的任务的过拟合。



​		本文在四个数据集上进行对比实验（CIFAR-10, CIFAR-100, SVHN与ImageNet），本模型比现有的算法有更少的参数，并且准确性优异。

## 3.相关工作

- **全连接层级联结构**。 一个级联机构在二十世纪八十代被研究。但开创性工作是集中在**以分层方式训练的全连接层的多层感知器上**。最近的基于全连接层的级联结构被提出，在小样本数据集上表现优秀，但是只适用于几百个参数的网络层。
- 跳跃连接。 通过**跳跃结构**使用CNN结构中的多级特征。导出了具有类似于本文中的的跨层连接的网络的纯理论框架。
- 使用旁路和门单元，例如Highway Networks。可端到端地训练100多层。
- 随机深度。ResNet使用**随机深度**方法训练1202层结构，**随机深度就是随机丢弃任意深度的网络结构。**
- Inception module。GoogLeNet用不同尺寸的过滤器制作的级联特征图。

## 4.DenseNets详解

### 4.1  神经网络结构

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423163440540.png)

DenseNet有三个dense blocks，中间使用convs与池化层进行连接，用于改变特征映射的尺寸。

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/2019042316592984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

growth rate：k = 32， “conv” layer包含BN-ReLU-Conv。

#### 4.1.1 Dense connectivity

第l层的输出为：
xl = Hl([x0, x1, . . . , xl-1])
Hl(·)代表非线性方程。
将l层之间的所有输出进行矩阵拼接，作为输入进行计算。

#### 4.1.2 Composite function

Hl(·)是符合方程，由三个连续的操作组成：

- batch normalization；
- ReLU；
- 3 × 3 convolution。

#### 4.1.3 Pooling layers

​		用于改变feature map的尺寸。卷积神经网络的一个改变feature map的重要部分就是**下采样**，在本文中，为了使用下采样，将整个神经网络分为3个模块。而模块之间的卷积层与池化层被称为 transition layers。

transition layers：

- batch normalization layer；
- 1×1 convolutional layer；
- 2×2 average pooling layer

#### 4.1.4 Growth ratel

​		若每一层都输出k个feature map，那么第l层有k0 +k ×(l−1)个feature map输入。k0代表通道数。通常k = 12。k就growth rate。
​		DenseNet与传统神经网络最大的不同就是，DenseNet可以只有很窄的网络层。

#### 4.1.5 Bottleneck layers

​		由于每一层都输出k个feature map，因此输入的feature map就会非常多，因此需要用到Bottleneck layers。

​		在每一个3×3 convolution之间加上1×1 convolution，用于**减少feature map的输入。**

​		dense blocks结构变成 BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)，也被称为 DenseNet-B，在本实验中，1×1 convolution输出4k feature-maps。

#### 4.1.6 Compression

​		在transition layers进一步压缩feature map的数量。将数量压缩θ倍，0 <θ ≤1。也被称为DenseNet-C。在本实验中，设置θ = 0.5 。

​		若bottleneck and transition layers均被使用，则被称为DenseNet-BC。

## 5.实验

在CIFAR、SVHN、ImageNet三个数据集上进行对比实验：

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423171353944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

图中代表错误率。
所有**没使用数据增强的DenseNets都使用dropout获得**。

- 当L = 190，k = 40时，在CIFAR数据集上的表现超出现有算法。
- 当L = 100，k = 24时，在SVHN数据集上的表现超出现有算法。
- 当DenseNet-BC网络层数超出250时，表现不会有很大增长。甚至会造成过拟合。

DenseNets对比实验：

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423171910350.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

通常当L与k增大时，算法的表现更优。当DenseNet-BC网络层数超出250时，表现不会有很大增长。甚至会造成过拟合。

DenseNets 与 ResNets top-1 error rates的对比实验：

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423172132567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423173252511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

​		DenseNets的参数比ResNets更少，DenseNets在transition layers使用到了 bottleneck structure与dimension reduction，从而参数更有效。
250-layer model 仅仅含有15.3M个参数。

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423173327833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/20190423180133607.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

![å¨è¿éæå¥å¾çæè¿°](https://img-blog.csdnimg.cn/2019042318162821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dvZHVpdGFvZG9uZzI2OTg=,size_16,color_FFFFFF,t_70)

像素(s，l)的颜色编码dense block内连接卷积层s到l的加权平均l1范数(归一化输入特征映射的数量)。

## 6.结论
DenseNets有更少的参数以及更少的计算。

Densenet集成了恒等映射、深度监督和多样化深度的特性。由于其紧凑的内部表示和减少的特征冗余，Densenet可能是一个很好的**特征提取器**。

