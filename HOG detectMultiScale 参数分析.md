# HOG detectMultiScale 参数分析

当我们用训练好的模型去检测测试图像时，我们会用到detectMultiScale() 这个函数来对图像进行多尺度检测。

这是opencv3.1里的参数解释

![img](https://images2015.cnblogs.com/blog/995848/201608/995848-20160808015157200-1576790123.png)

可以看到一共有8个参数。

1.img(必需)

这个不用多解释，显然是要输入的图像。图像可以是彩色也可以是灰度的。

2.foundLocations

存取检测到的目标位置

3.hitThreshold (可选)

opencv documents的解释是特征到SVM超平面的距离的阈值(*Threshold for the distance between features and SVM classifying plane)*

所以说这个参数可能是控制HOG特征与SVM最优超平面间的最大距离，当距离小于阈值时则判定为目标。

4.winStride(可选)

HoG检测窗口移动时的步长(水平及竖直)。

winStride和scale都是比较重要的参数，需要合理的设置。一个合适参数能够大大提升检测精确度，同时也不会使检测时间太长。

5.padding(可选)

在原图外围添加像素，作者在原文中提到，适当的pad可以提高检测的准确率（可能pad后能检测到边角的目标？）

常见的pad size 有*(8, 8)*, *(16, 16)*, *(24, 24)*, *(32, 32)*.

6.scale(可选)

**![img](https://images2015.cnblogs.com/blog/995848/201608/995848-20160808021941231-1564492297.png)**

如图是一个图像金字塔，也就是图像的多尺度表示。每层图像都被缩小尺寸并用gaussian平滑。

scale参数可以具体控制金字塔的层数，参数越小，层数越多，检测时间也长。 一下分别是1.01  1.5 1.03 时检测到的目标。 通常scale在1.01-1.5这个区间

![img](https://images2015.cnblogs.com/blog/995848/201608/995848-20160808024005293-998243228.png)![img](https://images2015.cnblogs.com/blog/995848/201608/995848-20160808024117793-436038047.png)![img](https://images2015.cnblogs.com/blog/995848/201608/995848-20160808024204512-1827448398.png)

7.finalThreshold（可选）

这个参数不太清楚，有人说是为了优化最后的bounding box

8.useMeanShiftGrouping(可选)

bool 类型，决定是否应用meanshift 来消除重叠。

default为false，通常也设为false，另行应用non-maxima supperssion效果更好。