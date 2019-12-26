[TOC]

# Faster R-CNN 代码分析（VGG16)

## 1 概述

​		在目标检测领域, Faster R-CNN表现出了极强的生命力, 虽然是2015年的[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1506.01497), 但它至今仍是许多目标检测算法的基础，这在日新月异的深度学习领域十分难得。Faster R-CNN还被应用到更多的领域中, 比如人体关键点检测、目标追踪、 实例分割还有图像描述等。

​		现在很多优秀的Faster R-CNN博客大都是针对论文讲解，本文将尝试从编程角度讲解Faster R-CNN的实现。由于Faster R-CNN流程复杂，符号较多，容易混淆，本文以VGG16为例，所有插图、数值皆是基于VGG16+VOC2007 。

### 1.1 目标

​		从编程实现角度角度来讲, 以Faster R-CNN为代表的Object Detection任务，可以描述成:

给定一张图片, 找出图中的有哪些**对象**,以及这些对象的**位置**和置信**概率**。

![img](https://pic2.zhimg.com/80/v2-79ee2d46ba80b773089056b69b55b991_hd.jpg) 																目标检测任务

### 1.2 整体架构

Faster R-CNN的整体流程如下图所示。

![img](https://pic3.zhimg.com/80/v2-4e372e4536ef6d3d28ebd8803a9b13e2_hd.jpg)														Faster R-CNN整体架构

从编程角度来说， Faster R-CNN主要分为四部分（图中四个绿色框）：

- Dataset：数据，提供符合要求的数据格式（目前常用数据集是VOC和COCO）
- Extractor： 利用CNN提取图片特征`features`（原始论文用的是ZF和VGG16，后来人们又用ResNet101）
- RPN(*Region Proposal Network):* 负责提供候选区域`rois`（每张图给出大概2000个候选框）
- RoIHead： 负责对`rois`分类和微调。对RPN找出的`rois`，判断它是否包含目标，并修正框的位置和座标

Faster R-CNN整体的流程可以分为三步：

- 提特征： 图片（`img`）经过预训练的网络（`Extractor`），提取到了图片的特征（`feature`）
- Region Proposal： 利用提取的特征（`feature`），经过RPN网络，找出一定数量的`rois`（region of interests）。
- 分类与回归：将`rois`和图像特征`features`，输入到`RoIHead`，对这些`rois`进行分类，判断都属于什么类别，同时对这些`rois`的位置进行微调。

## 2 详细实现

### 2.1 数据

​		对于每张图片，需要进行如下数据处理：

- 图片进行缩放，使得长边小于等于1000，短边小于等于600（至少有一个等于）。
- 对相应的bounding boxes 也进行同等尺度的缩放。
- 对于Caffe 的VGG16 预训练模型，需要图片位于0-255，BGR格式，并减去一个均值，使得图片像素的均值为0。

最后返回四个值供模型训练：

- images ： 3×H×W ，BGR三通道，宽W，高H

- bboxes： 4×K , K个bounding boxes，每个bounding box的左上角和右下角的座标，形如（Y_min,X_min, Y_max,X_max）,第Y行，第X列。

- labels：K， 对应K个bounding boxes的label（对于VOC取值范围为[0-19]）

- scale: 缩放的倍数, 原图H' ×W'被resize到了HxW（scale=H/H' ）

  ​		需要注意的是，目前大多数Faster R-CNN实现都只支持batch-size=1的训练（[这个](https://zhuanlan.zhihu.com/github.com/jwyang/faster-rcnn.pytorch) 和[这个](https://link.zhihu.com/?target=https%3A//github.com/precedenceguo/mx-rcnn)实现支持batch_size>1）。

###  2.2 Extractor

​		Extractor使用的是**预训练好**的模型提取图片的特征。论文中主要使用的是Caffe的预训练模型VGG16。

​		修改如下图所示：为了节省显存，前四层卷积层的学习率设为0。Conv5_3的输出作为图片特征（feature）。conv5_3相比于输入，下采样了16倍，也就是说输入的图片尺寸为3×H×W，那么`feature`的尺寸就是C×(H/16)×(W/16)。

​		VGG最后的三层全连接层的前两层，一般用来初始化RoIHead的部分参数，这个我们稍后再讲。总之，一张图片，经过extractor之后，会得到一个C×(H/16)×(W/16)的feature map。

![img](https://pic4.zhimg.com/80/v2-28887eb4f69439e1384165da0ca20b6f_hd.jpg)

​												Extractor: VGG16



### **2.3 RPN**

​		Faster R-CNN最突出的贡献就在于提出了Region Proposal Network（RPN）代替了Selective Search，从而将候选区域提取的时间开销几乎降为0（2s -> 0.01s）。

#### **2.3.1 Anchor**

​		在RPN中，作者提出了`anchor`。**Anchor是大小和尺寸固定的候选框。**论文中用到的anchor有三种尺寸和三种比例，如下图所示，三种尺寸分别是小（蓝128）中（红256）大（绿512），三个比例分别是1:1，1:2，2:1。3×3的组合总共有9种anchor。



![img](https://pic4.zhimg.com/80/v2-7abead97efcc46a3ee5b030a2151643f_hd.jpg)

​																		Anchor



​		然后用这9种anchor在特征图（`feature`）左右上下移动，每一个特征图上的点都有9个anchor，最终生成了 (H/16)× (W/16)×9个`anchor`. 对于一个512×62×37的feature map，有 62×37×9~ 20000个anchor。 也就是对一张图片，有20000个左右的anchor。这种做法很像是暴力穷举，20000多个anchor，哪怕是蒙也能够把绝大多数的ground truth bounding boxes蒙中。

**2.3.2 训练RPN**

RPN的总体架构如下图所示：

![img](https://pic3.zhimg.com/80/v2-e7eeb94a86ece2dadfa9db2277f7d016_hd.jpg)RPN架构

anchor的数量和feature map相关，不同的feature map对应的anchor数量也不一样。RPN在`Extractor`输出的feature maps的基础之上，先增加了一个卷积（用来语义空间转换？），然后利用两个1x1的卷积分别进行二分类（是否为正样本）和位置回归。进行分类的卷积核通道数为9×2（9个anchor，每个anchor二分类，使用交叉熵损失），进行回归的卷积核通道数为9×4（9个anchor，每个anchor有4个位置参数）。RPN是一个全卷积网络（fully convolutional network），这样对输入图片的尺寸就没有要求了。

接下来RPN做的事情就是利用（`AnchorTargetCreator`）将20000多个候选的anchor选出256个anchor进行分类和回归位置。选择过程如下：

- 对于每一个ground truth bounding box (`gt_bbox`)，选择和它重叠度（IoU）最高的一个anchor作为正样本
- 对于剩下的anchor，从中选择和任意一个`gt_bbox`重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
- 随机选择和`gt_bbox`重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。

对于每个anchor, gt_label 要么为1（前景），要么为0（背景），而gt_loc则是由4个位置参数(tx,ty,tw,th)组成，这样比直接回归座标更好。

![[公式]](https://www.zhihu.com/equation?tex=t_x+%3D+%28x+%E2%88%92+x_a%29%2Fw_a%3B+t_y+%3D+%28y+%E2%88%92+y_a%29%2Fh_a%3B%5C%5C+t_w+%3D+log%28w%2Fw_a%29%3B+t_h+%3D+log%28h%2Fh_a%29%3B%5C%5C+t_x%5E%2A+%3D+%28x%5E%2A+%E2%88%92+x_a%29%2Fw_a%3B+t_y%5E%2A+%3D+%28y%5E%2A+%E2%88%92+y_a%29%2Fh_a%3B%5C%5C+t_w%5E%2A+%3D+log%28w%5E%2A%2Fw_a%29%3B+t_h%5E%2A+%3D+log%28h%5E%2A%2Fh_a%29%3B%5C%5C)

计算分类损失用的是交叉熵损失，而计算回归损失用的是Smooth_l1_loss. 在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。

**2.3.3 RPN生成RoIs**

RPN在自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(`ProposalCreator`)如下：

- 对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。
- 选取概率较大的12000个anchor
- 利用回归的位置参数，修正这12000个anchor的位置，得到RoIs
- 利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs

注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.

注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。

RPN的输出：RoIs（形如2000×4或者300×4的tensor）

## **2.4 RoIHead/Fast R-CNN**

RPN只是给出了2000个候选框，RoI Head在给出的2000候选框之上继续进行分类和位置参数的回归。

**2.4.1 网络结构**

![img](https://pic3.zhimg.com/80/v2-5b0d1ca6e990fcdecd41280b69cd8622_hd.jpg)RoIHead网络结构



由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。首先利用`ProposalTargetCreator` 挑选出128个sample_rois, 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。下图就是一个例子，对于feature map上两个不同尺度的RoI，经过RoIPooling之后，最后得到了3×3的feature map.

![img](https://pic4.zhimg.com/80/v2-d9eb14da175f7ae2ed6b6d77f8993207_hd.jpg)RoIPooling

RoI Pooling 是一种特殊的Pooling操作，给定一张图片的Feature map (512×H/16×W/16) ，和128个候选区域的座标（128×4），RoI Pooling将这些区域统一下采样到 （512×7×7），就得到了128×512×7×7的向量。可以看成是一个batch-size=128，通道数为512，7×7的feature map。

为什么要pooling成7×7的尺度？是为了能够共享权重。在之前讲过，除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG16预训练的权重，初始化前两层全连接。最后再接两个全连接层，分别是：

- FC 21 用来分类，预测RoIs属于哪个类别（20个类+背景）
- FC 84 用来回归位置（21个类，每个类都有4个位置参数）

**2.4.2 训练**

前面讲过，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用`ProposalTargetCreator` 选择128个RoIs用以训练。选择的规则如下：

- RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
- 选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本

为了便于训练，对选择出的128个RoIs，还对他们的`gt_roi_loc` 进行标准化处理（减去均值除以标准差）

对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.而且是只对正样本中的这个类别4个参数计算损失。举例来说:

- 一个RoI在经过FC 84后会输出一个84维的loc 向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss
- 如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失。

**2.4.3 生成预测结果**

测试的时候对所有的RoIs（大概300个左右) 计算概率，并利用位置参数调整预测候选框的位置。然后再用一遍极大值抑制（之前在RPN的`ProposalCreator`用过）。

注意：

- 在RPN的时候，已经对anchor做了一遍NMS，在RCNN测试的时候，还要再做一遍
- 在RPN的时候，已经对anchor的位置做了回归调整，在RCNN阶段还要对RoI再做一遍
- 在RPN阶段分类是二分类，而Fast RCNN阶段是21分类

## **2.5 模型架构图**

最后整体的模型架构图如下：

![img](https://pic3.zhimg.com/80/v2-7c388ef5376e1057785e2f93b79df0f6_hd.jpg)整体网络结构

需要注意的是： 蓝色箭头的线代表着计算图，梯度反向传播会经过。而红色部分的线不需要进行反向传播（论文了中提到了`ProposalCreator`生成RoIs的过程也能进行反向传播，但需要专门的[算法](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1512.04412)）。