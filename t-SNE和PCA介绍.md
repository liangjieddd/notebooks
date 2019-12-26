[TOC]

# t-SNE和PCA介绍

## 1.t-SNE

1. t-SNE : t-分布领域嵌入算法，读作“Tee-Snee”，它只在用于已标记数据时才真正有意义，可以明确显示出**输入的聚类状况。**
    主要想法就是，**将高维分布点的距离，用条件概率来表示相似性，同时低维分布的点也这样表示。**
    只要二者的**条件概率非常接近**（用相对熵来训练，所以需要label），那就说明**高维分布的点已经映射到低维分布上**了。
2. 难点：高维距离较近的点，比较方便聚在一起，但是高维距离较远的点，却比较难在低维拉开距离。
    其次，训练的时间也比较长
3. 建议观赏链接，绝对牛逼的t-SNE介绍：[从SNE到t-SNE再到LargeVis

![img](https://upload-images.jianshu.io/upload_images/12678831-f24403923ca21dc0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

## 2.PCA

1. PCA（Principal Component Analysis）主要成分分析。
    PCA把**原先的n个特征用数目更少的m个特征取代**，新特征是**旧特征的线性组合**，这些线性组合**最大化样本方差**，尽量使新的m个特征互不相关。从旧特征到新特征的映射捕获数据中的固有变异性。
    不仅仅是对高维数据进行**降维**，更重要的是经过降维**去除了噪声**，发现了数据中的模式。
2. 计算过程：

- 原始数据进行**特征均值化**
- 计算特征均值化后的**协方差矩阵**（算出特征之间的关系）
- 计算协方差矩阵的**特征值和特征向量**（特征值分解）
- 选取**大的特征值**对于的特征向量来更新原始数据集（直接相乘就好）

1. PCA涉及协方差，协方差（conv）:
    方差的定义：

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-58a3785cf290ed55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/241/format/webp)

   

    即，**度量各个维度偏离均值的程度**。仿照其，协方差的定义：

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-34ec50ec8383896b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/258/format/webp)

   

    假设我们想统计一个男孩子的猥琐程度跟他受女孩子的欢迎程度是否存在一些联系，这是个二维的特征问题，我们用协方差来计算之间的联系。**协方差的结果如果为正值，则说明两者是正相关的**（从协方差可以引出“相关系数”的定义），也就是说一个人越猥琐越受女孩欢迎。

   **如果结果为负值， 就说明两者是负相关，**越猥琐女孩子越讨厌。

   **如果为0，则两者之间没有关系，**猥琐不猥琐和女孩子喜不喜欢之间没有关联，就是统计上说的“相互独立”。

    从协方差的定义上我们也可以看出一些显而易见的性质，如：

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-298c65c79359e4f5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/176/format/webp)

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-f5926fa2ee85dc37.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/195/format/webp)

   

2. 协方差矩阵
    协方差只能处理二维问题，维数一多，自然需要计算多个协方差，由此需要矩阵来组织。协方差矩阵定义：

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-1502791b25872d7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/309/format/webp)

   

   对于n维的数据集要算协方差，得到的协方差矩阵大小就为n^2。但是实际计算次数（每次不分次序抽两个）只需要

   

   ![img](https:////upload-images.jianshu.io/upload_images/12678831-ef359174fbad9097.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/91/format/webp)

   

    可见，协方差矩阵为对称的矩阵，对角线又为各个维度的方差。

 5. 观赏链接 :  [主成分分析PCA](https://www.cnblogs.com/zhangchaoyang/articles/2222048.html)
 



![img](https:////upload-images.jianshu.io/upload_images/12678831-b47c5132490f60ac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)