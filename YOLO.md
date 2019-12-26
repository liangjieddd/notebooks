[TOC]

# YOLO

# 1. 引言

​		目标检测算法是计算机视觉三大基础任务之一，其包括目标定位和目标分类两部分。

​		在 yolo 系列出来之前，主流的做法是分段式的 R-CNN 系列，主要包括 R-CNN、Fast R-CNN、Faster R-CNN、Mask R-CNN 等。

# 2. R-CNN系列

## 2.1 R-CNN 基本结构和原理

R-CNN 的基本结构如下图所示：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS499uiamGZRoBbTbLq1KQzdDupyj33VZc3clMUZhZB5l1nvOPN2VDib57A/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

​		R-CNN 主要分为候选区提取和候选区分类两个阶段，并且两阶段分开训练。其主要思想如下。

​		首先通过选择性搜索（Selective Search）对输入进行超像素合并，产生基础的子区域。然后将小的子区域不断合并成大区域，并从中找出可能存在物体的区域，这个过程即候选区提取（Region Proposal）。	

​		提取出包含目标的候选区之后，需要对其进行分类，判定目标属于哪一类。可以通过 SVM 或 CNN 等算法进行分类。

## 2.2 R-CNN 的不足与改进

### **2.2.1 SPP 和 ROI**

​		要实现较为实用的 R-CNN 网络，往往需要对输入样张提取上千个候选区，并对每个候选区进行一次分类运算。于是，后续出现空间金字塔池化（SPP） 和 region of interest（ROI）等方式进行改进。

​		其基本思想是，输入图片中的目标区域，经过 CNN 后，得到的特征图中，往往也存在着对应的目标区域，此即 ROI。后续对该特征图（多种尺度）上的 ROI 进行分类，此即 SPP。

​		通过这种方式，可以共用特征提取部分，只对最后的特征图进行候选区提取和分类。这样就可以极大地减少总的计算量，并提升性能。

### 2.2.2 **Fast R-CNN**

​		但是，SPP 和 ROI 方式，仍旧需要分段训练。其不仅麻烦，同时还分割了 bounding box 回归训练与分类网络的训练。这使得整个函数的优化过程不一致，从而限制了更高精度的可能。

于是，再次对其进行改进：

1.进行 ROI 特征提取之后，将两种损失进行合并，统一训练。这样相对易于训练，且效率更高

2.将 SPP 换做 ROI Pooling

3.对于 bounding box 部分的 loss，使用 Smooth l1 函数，可以提升稳定性

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4OhNVMCMiagaB6URePFBicdVb9FYPXf75cfoIicGmKicAutxd5lcsrePfTw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.2.3 **Faster R-CNN**

​		在 Fast R-CNN 中，对 ROI 而非原图的候选区进行分类，提升速度。因此，下一步可以对候选区提取部分进行优化，提升性能。

​		因此，在 Faster R-CNN 中，不再对原图进行候选区提取，而是直接对经过 CNN 后的特征图进行候选区提取，这部分网络，即 Region Proposal Networks（RPN）。

​		之后，将候选区分别送入两个子网络，分别用与计算 bounding box 的坐标和候选区的类别。如图 1 所示。

通过这种方式，可以进一步减少计算量，合并两个阶段，并提升精度。

# 3. YOLO

### 3.1 YOLO V1

#### 3.1.1 主要贡献和优势

​		虽然 Fast R-CNN 已经相当优秀，但是其仍旧不是真正意义上的一体式目标检测网络，其性能仍有提升的空间。

​		针对 R-CNN 系列的分段式设计的问题，YOLO 提出一种全新的 loss 计算方式，重新定义了目标检测问题，将其定义为回归问题进行训练，同时进行定位和分类的学习。

​		YOLO 的核心，在于其**损失函数**的设计。其一体式架构设计，计算量更少，速度更快，易于优化，且满足实时检测的需求。



**YOLO V1 具有以下优势：**

1.速度极快，易于优化：只需读取一次图像，就可进行端对端优化，可满足实时需求

2.背景误识别率低：对全图进行卷积学习，综合考虑了全图的上下文信息

3.泛化性能好：也是由于综合考虑了图片全局，因此能够更好地学习数据集的本质表达，泛化性能更好

4.识别精度高



当然，相较于 Faster R-CNN ，YOLO v1 存在**明显不足**：

1.定位精度不够，尤其是小目标

2.对密集目标的识别存在不足

3.异常宽长比的目标识别不佳



#### 3.1.2基本原理

> 深度学习任务中，合理的目标设定，是成功的关键因素之一。

##### 3.1.2.1 **Anchor box 的设计**

​		在 R-CNN 系列中，需要先提取候选区，然后再将候选区进行回归微调，使之更接近 groung truth。

​		而 YOLO 直接将其合并为一步，即：回归。但是，YOLO 保留了候选区的思想，只是将其演变为了 anchor box。

​		在 YOLO V1 中，首先设定 B 个不同尺寸，宽长比的 anchor box。然后将每张图片划分成S×S的格点，每个格点对应有 B 个 anchor box，共S×S×B个 anchor box，其粗略的覆盖了整张图片。

​		对于每张图片，均存在初始的S×S×B 个 anchor box，作为初始 bounding box。现在需要做的是，通过学习，不断判定哪些 bounding box 内存在目标，存在什么样的目标，同时不断调整可能存在目标的 bounding box 的宽长比和尺寸，使之与 ground truth 更接近。

那么，ground truth 又是如何定义的呢？

##### 3.1.2.2 **Ground truth 的生成**

​		目标检测任务，首先需要做的是判定是否包含目标，然后才是判定目标的位置以及类别。

以下图为例，详细讲解从一张图片，生成 ground truth 的过程。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4vShcH5TRibNpPeckCbxpuWDbJRJ9xj6ou9FbzY8Xc19O7hE2duXLLeQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**confidence score**

​		如上图所示，首先要做的，就是判定哪些 bounding box 内包含目标。我们可以通过一个置信度得分 confidence score，来判定某个 bounding box 内是否包含目标。

​		对于上图，我们设定目标中心所在的格点，包含目标。显然，对于图片中目标的 ground truth，该值为 1。



**坐标值换算**

​		对于不包含目标的 anchor box，不用计算其坐标。对于包含目标的 anchor box，需要获取其坐标。

​		在 yolo 中，通过**中心位置和尺寸**表示坐标，即：(x,y,w,h)

​	

图片的 label 与 groung truth 之间，通过如下方式换算。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4HuRwoTtHKf1SfFZBD2WapWUnhibpuiasuzbnwQLqwcIKphRAFFO8mEgw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

如上图所示，狗狗的原始坐标（label）为![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4F574NGadXn4WpMicOMnfnQdIX4B11rZVnpYXuhphHhPZrzfGlAicAsaA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。现在需要将其变换为 ground truth。

 		x,y表示目标中心距离对应格点边界（左上角）的位置，数值相对于格点单元尺寸进行过归一化。w,h为目标边框的尺寸，相对于整图尺寸进行过归一化。

​		因此，上图狗狗的坐标对应的 ground truth 计算方式为（每个格点尺寸为 1）：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4LicG0IQk5wa1FMib03qO8E4GxiaPJs8RE89YRgmNL2LzJUiaGZWKyxRyAQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**类别概率**

​		目标位置定义完毕后，需要定义目标的类别对应的 ground truth。在 YOLO V1 中，存在如下设定：

​		**每个格点最多只能预测一个目标，**即使是共有 B 个 anchor box，因此最多只能包含一个类别。

​		这一设定，将会导致，对于密集的目标，YOLO 的表现较差。

​		对于每个目标（每个格点内只允许存在一个），其对应的类别 ground truth 为 one-hot 编码。



**映射到 bounding box**

​		由于每张图片初始对应S×S×B个 bounding box，因此需要将上面的 ground truth 进行映射，使之与 bounding box 尺寸对应一致，才能进行比较，计算误差，进行优化。步骤如下：

1.初始化每个格点上的 bounding box 为 0

2.对于存在目标的格点，将于 ground truth 之间 IOU 最大的 bounding box 对应的 confidence score 填充为 1，表示包含目标

3.将包含目标的 bounding box ，填充对应的的 ground box 的坐标和类别值

​		到这里，就从 label ，得到了用于比较的 target。

##### 3.1.2.3 **推理过程**

​		推理过程较为简单，输入图片，得到一个尺寸为 S×S的特征图，通道数为B×（1+4）+C，其中，B 为每个格点的 bounding box 数目，C 为预测的目标类别数，1 和 4 分别表示包含目标的置信度和对应的坐标。

​		由于每个格点只负责预测一个目标，因此只需要包含一个类别向量即可。

​		在原论文中，S=7，B=2，C=20，因此最后输出尺寸为7×7×30 。



##### 3.1.2.4 **计算 loss**

​		对于每张图片，大多数格点单元不包含目标，其对应的置信度得分为 0。这种目标存在与否的失衡，将会影响最后 loss 的计算，从而影响包含目标的格点单元的梯度，导致模型不稳定，训练容易过早收敛。

​		因此，我们增加 bounding box 坐标对应的 loss，同时对于不包含目标的 box，降低其置信度对应的 loss。

​		我们用![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4siaV0KasjCWGdeefgvB0KEkKGGia0wga3GuRAXvkqK94wQGuJVkWfLOA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS40T0ibz2yZX576Ngq1cQpZuOg8sxFjXKH4V8YTCBx45wfGtcnwnGUxsg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)来实现这一功能，且：![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4ajw7TKhmxjFYdx4jCCK2f3jj5kD42zk3kzpVE7a6p6WrJwTWeqWIbw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。

​		同时，sum-squared error 还会同等看待 large boxes 和 small boxes 的 loss 。而同等的 loss 对于 large boxes 和 small boxes 的影响是不同的。

​		为了减缓这种空间上的不均衡，我们选择预测 w 和 h 的平方根，可以降低这种敏感度的差异，使得较大的对象和较小的对象在尺寸误差上有相似的权重。

​		综上所述，完整的 loss 计算方式如下所示：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4p6gIyPIuqlwJIicexAibsrJO6c5EQbraVtmYqHaAaMGzOydhwGjwXTpA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其中（以 ground truth 为判定依据）：



•![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4blfia4ffyRJ5XKyd1mmYr8vVSlLT6JhaeiapltLWnFCeME6ATqcvhUMw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)表示是否格点单元i中包含目标；



•![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4mGYyQ9YibqJwXkgL71va7niaQCwJsPVYSdfkfLjMlC56RLAt8hXviboyw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)表示格点单元i中，第j个预测的 bounding box 包含目标



•![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4fkjGTlhia09OeXTmM4Sibsg8ibSfo46uQ0h7AIQe9mwwtN6SibKze4AmDQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)意思是网格 i 的第 j 个 bounding box 中不存在对象



因此，上面的 loss 中：

• 第一行表示：当第 i 个格点中第 j 个 box 中存在目标 (IOU 比较大的 bounding box) 时，其坐标误差

• 第二行表示：第 i 个格点中第 j 个 box 中存在目标时，其尺寸误差

• 第三行表示：第 i 个格点中第 j 个 box 中存在目标时，其置信度误差

• 第四行表示：第 i 个格点中第 j 个 box 中不存在目标时，其置信度误差

• 第五行表示：第 i 个格点中存在目标时，其类别判定误差



##### 3.1.2.5 实用过程

​		在实际使用中，需要预测实际的边框和类别。通常，可能存在多个 bounding box 预测一个目标，存在冗余，需要使用非极大抑制（NMS）来剔除冗余 bounding box。

​		其核心思想是：选择置信度得分最高的作为输出，去掉与该输出重叠较高的预测框，不断重复这一过程直到处理完所有备选框（共  S×S×B个）。

**具体步骤如下所示：**

1. 过滤掉 confidence score 低于阈值的 bounding box

2. 遍历每一个类别
   1. 找到置信度最高的 bounding box，将其移动到输出列表
   2. 对每个 Score 不为 0 的候选对象，计算其与上面输出对象的 bounding box 的 IOU
   3. 根据预先设置的 IOU 阈值，所有高于该阈值（重叠度较高）的候选对象排除掉
   4. 当剩余列表为 Null 时， 则表示该类别删选完毕，继续下一个类别的 NMS

3. 输出列表即为预测的对象

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4xpEAqsvc0nyQbvZ11MMJDFlcfLAH0zCd7iaFicKfZq2Lpr3LPfZI8JZA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

网络结构

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS41PicyGLAXwDkqvEM1wRRrByCk2lhFj0LHHibedgY7AdSj6KLcRKJf6Zg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



### 3.2 Yolo V2

**Yolo V2 的主要贡献在于：**

1. 利用 word Tree 设计，充分利用分类数据集，弥补目标识别类别数目的不足

2. 重新设计基础网络 darknet-19，输入尺寸可变，从而在同一套模型上，提供速度和精度之间的切换

3. 重新设计 anchor box 和坐标变换格式，使的收敛更快，精度更高



**关键改进**：

​		这篇论文进行了较多的改进优化，主要分为新设计的基础网络 darknet-19，以及新设计 anchor box 等。至于其他改进，详见论文。



**重新定义 Anchor box:**

​		在 Yolo V2 中，输入尺寸变为416×416，网络整体缩放倍数为 13，最后得到尺寸为13×13的特征图，并在该尺寸上进行推理预测。

​		此外，较为重要的是，v2 中，每个 bounding box 负责预测一个目标，因此一个格点内可以预测多个目标，解决了密集目标的预测问题。	

​		此外，不再通过手工选择 anchor box，而是针对特定数据集，通过 k-means 算法进行选择，详见论文。



**坐标变换方式：**

​		在 yolo v1 中，使用全连接层来直接预测目标的 bounding box 的坐标。训练过程不够稳定，主要来自（x,y）的预测。

​		而在 Faster R-CNN 中，使用全卷积网络 RPN 来预测 bounding box 相对于 anchor box 的坐标的偏移量。由于预测网络是卷积网络，因此 PRN 在 feature map 网络的每个位置预测这些 offset。

​		相比于直接预测坐标，预测 offset 更简单，误差更小，可以简化问题，使得网络更容易学习。

​		原始方案中，预测值![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4p5JpD7akLA30prhdGZqFuUF9JuKh2uVbgdEzBvNbcJ47ibeOgYwTszw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)和(x,y)之间，计算方式如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4rmoAI1ugBymc2hNVGO2fzzSyfL0kUkMKrQEfEia1CgXYwjpP0juwLSA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

​		该方式下，对坐标![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4p5JpD7akLA30prhdGZqFuUF9JuKh2uVbgdEzBvNbcJ47ibeOgYwTszw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)没有限制，因此预测的 bounding box 可能出现在任意位置，导致训练不稳定。因此，在 V2 内改为预测偏移量，其计算方式如下所示:

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4X26Kn6oLrF6SoRF2ya3x4iarcDMaUyiaumjxFMEPZdH5cMdzrK4SicFJw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

其中，![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4URfib52qbMJIoibH9zrQewSv6ibySNJR9ibrWibPEhGvicxsDpiccx5icczC4Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)表示格点单元相对于图像左上角的坐标；![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4vXdVXG2HOegvUU2icJ5xBcSjQMH3dMaAsL0wl5uich7nn2sDgkc9OUkw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)表示先验框的尺寸 (bounding box prior)，预测值为![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4GUNmcRzrDSJrpFn5lE7RRBwTXEyiaPwWsDeRKS9QSDTTIMrL2vIh5BA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)。



- 对于预测的 bbox 的中心，需要压缩到 0-1 之间，再加上 anchor 相对于grid 在 x 和 y 方向上的偏移。这一点，和 yolo v1 是一致的
- 对于预测的 bbox 的宽高，这个和 faster RCNN 一样，是相对于 anchor 宽高的一个放缩。exp(w) 和 exp(h) 分别对应了宽高的放缩因子
- 对于预测的 bbox 的置信度，则需要用 sigmoid 压缩到 0-1 之间。这个很合理，因为置信度就是要0-1之间。
- 对于预测的每个类别，也是用你 sigmoid 压缩到0-1之间。这是因为类别概率是在0-1之间

​		最后通过换算得到的为在当前特征图尺寸上的坐标和尺寸，需要乘以整体缩放因子(32)，方可得到在原图中的坐标和尺寸。

这种参数化的方式，使得神经网络更加稳定。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4aVcyCIib5bBiapMWia3ybWD1dcw2gR8M9Z6b0iaRUgc8AAGszcCXngqzQA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

**多尺度融合：**

​		13×13的输出特征图，可以很好的预测较大尺寸的目标，但是对于小尺寸的目标，可能并不太好。

​		因此，在 YOLO v2 中，除了使用13×13的特征图，还使用其之前层尺寸为26×26和52×52的特征图，并进行多尺度融合。不同尺寸之间，通过如下形式，进行特征融合。

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4UZsG3myryTyvU5aANvNqKQhKjoLwqcD4jDsPON06kTI45aNNFMXgxw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

例如，26×26×256通过这种方式，将变为13×13×1024的 tensor。

具体融合形式，详见图 9。



**Darknet-19**

​		大多数 detection 系统以 VGG-16 作为基础的特征提取器 (base feature extractor)。但是 vgg-16 较为复杂，计算量较大。

​		Yolo V2 使用一个定制的神经网络作为特征提取网络，它基于 Googlenet 架构，运算量更小，速度更快，然而其精度相较于 VGG-16 略微下降，称之为 darknet-19。

​		与 VGG 模型类似，大多数使用3×3的卷积层，并在每一次 pooling 后，通道数加倍。此外，使用全局池化 ( GAP，global average pooling ) 来进行预测。同时，在3×3的卷积层之间，使用1×1的卷积层来压缩特征表达。此外，使用 batch normalization 来稳定训练，加速收敛以及正则化模型。



完整的网络层如下所示：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS45PribRmVScAm1ic5E41AOY6Mnb97o2eWLcf50dvt6nliaicCTTxz6tukog/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

### 3.3 Yolo V3

​		Yolo V3 只是对 Yolo v2 进行了一次较小的优化，主要体现在网络结构上，提出了 darknet-53 结构，作为特征提取网络。最后，Yolo V3 在小目标的识别上改善较大，但是中等目标和大目标的识别方面，表现略微下降。

网络结构如下所示：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4ywOAKTekib0oAmkjpgdpmhhD4d6AYudqhAGgeAX9b7bEdtepQu8iaicKQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4fN1oKcQHEVNeRcpWK3IJRVRvIwdGFzLgP3JVWw4H7YG4oAoqnQHPjQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

最后附上一张性能表现图：

![img](https://mmbiz.qpic.cn/mmbiz_png/75DkJnThACk1gicdd2tc4qqdUWzsCibHS4qfv9dmMMcojJeSu4ZWcNb08py21ibMqwm8ClMibEJ2MFNAfSmOZgSxhQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)



# 4 参考

\- https://arxiv.org/abs/1506.02640

\- https://arxiv.org/pdf/1612.08242.pdf

\- https://pjreddie.com/media/files/papers/YOLOv3.pdf

\- https://pjreddie.com/darknet/yolo/

\- https://github.com/BobLiu20/YOLOv3_PyTorch

\- https://www.jianshu.com/p/d535a3825905