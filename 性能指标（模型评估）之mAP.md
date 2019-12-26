[TOC]

# 性能指标（模型评估）之mAP

## 1.性能指标

​		用于评价模型的好坏，当然使用不同的性能指标对模型进行评价往往会有不同的结果，也就是说模型的好坏是“相对”的，什么样的模型好的，不仅取决于算法和数据，还决定于任务需求。因此，选取一个合理的模型评价指标是非常有必要的。

## 2.错误率 & 精度
针对数据集D和学习器f而言：

1、错误率：分类错误的样本数占总样本的比例 
$$
E(f;D)=\frac 1m \sum_{i=1}^mI(f(x_i) \neq y_i)
$$
2、精度：分类正确的样本数占总样本的比例
$$
\begin{align}
acc(f;D) & =\frac 1m \sum_{i=1}^mI(f(x_i) = y_i) \\
& =1- E(f;D)
\end{align}
$$

## 3.召回率 & 准确率
​		精度和错误率虽然常用，但还是不能满足所有的需求。举个例子：

​		信息检索中，我们经常会关系“检索出的信息有多少比例是用户感兴趣的”以及“用户感兴趣的信息中有多少被检索出来了”，用精度和错误率就描述出来了，这就需要引入准确率（precision，亦称查准）和召回率（recall，亦称查全）。

![1562939478816](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1562939478816.png)

#### 准确率

预测结果中，究竟有多少是真的正？（找出来的对的比例）
$$
P=\frac {TP}{TP+FP}
$$

#### 召回率

所有正样本中，你究竟预测对了多少？（找回来了几个） 
$$
R=\frac {TP}{TP+FN}
$$

### P-R曲线

​		一般来说，我们希望上述两个指标都是越高越好，然而没有这么好的事情，准确率和召回率是一对矛盾的度量，一个高时另一个就会偏低，当然如果两个都低，那肯定时哪点除了问题。

![img](https://img-blog.csdn.net/20170826114737648?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​		当我们根据学习器的预测结果对样例进行排序（排在前面的时学习器认为“最可能”是正例的样本），然后按此顺序依次吧样本喂给学习器，我们把每次的准确率和召回率描出来就会得到一个P-R曲线（称为P-R图）。根据这个图怎么评估不同的学习器的好坏呢？

​		**直观感受**：如果一个学习器的P-R被另一个学习器的该曲线包围，则可以断言后面的要好些。
​		但是如果两个曲线有交叉，那就很难说清楚了。

​		一个比较合理的判据是我比较下两个曲线下面的面积大小，他能在一定程度上反应P和R“双高”的比例，但问题是这个面积值不太容易估算啊。那有没有综合考虑这两个指标的指标呢？当然是有的，且看下面

### 平衡点（Break-Even Point, BEP）
就是找一个 **准确率 = 召回率** 的值，就像上面的图那样。

### F1度量
F1是准确率和召回率的调和平均，即是
$$
\frac 1 {F1} =\frac 12 \times (\frac 1P+\frac 1R)
$$
换算下：
$$
F1=\frac {2PR}{P+R}
$$
然而，在更一般的情况下，我们对P和R的重视程度又是不同的，因此，F1度量的更一般的形式可以写作加权调和平均
$$
F_\beta
$$
即是
$$
\frac 1 {F_\beta} =\frac 1{1+\beta ^2} \times (\frac 1P+\frac {\beta ^2} R)
$$
换算下：
$$
F_\beta=\frac {(1+\beta ^2)PR}{\beta ^2P+R}
$$

## 4.mAP

### mAP是什么
​		多标签图像分类任务中图片的标签不止一个，因此评价不能用普通单标签图像分类的标准，即mean accuracy，该任务采用的是和信息检索中类似的方法—mAP（mean Average Precision），虽然其字面意思和mean accuracy看起来差不多，但是计算方法要繁琐得多。

### 计算过程

> 保存所有样本的 confidence score

​		首先用训练好的模型得到所有测试样本的confidence score，每一类（如car）的confidence score保存到一个文件中（如comp1_cls_test_car.txt）。假设共有20个测试样本，每个的id，confidence score和ground truth label如下：

![img](https://img-blog.csdn.net/20170826194003211?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

对confidence score进行排序:

![img](https://img-blog.csdn.net/20170826194118274?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

计算precision和recall

​		上面我们一共有20个测试样本，如果把这20个样本放在一起，按照表1给出的把他们分成4类，就可以得到下面的示意图：

![img](https://img-blog.csdn.net/20170826194344330?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​		其中，圆圈内（真正 + 假正）是我们模型预测为正的元素，比如对测试样本在训练好的car模型上分类（如果是car，输出label = 1，反之=0），现在假设我们想得到top-5的结果，也就是说圆圈内一共有5个数据，即排序好的表的前面5个：

![img](https://img-blog.csdn.net/20170826194456817?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

好了，上表就是我们预测为正的元素啦，他的准确率是多少？
$$
\begin{align}
P & =\frac {TP}{TP+FP} \\
& =\frac {2}{2+3} \\
& =\frac 25=40 \%
\end{align}
$$
召回率是多少呢？在这里请注意我们的所有测试样本一共有多少个car（也就是label=1有几条数据），在下表中很容易找到**6条记录**，那我们预测出来的结果找到几个car呢？上面的top-5中我们只找到了**2个car**。 

![img](https://img-blog.csdn.net/20170826194118274?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

也就是说，召回率为：
$$
\begin{align}
R & =\frac {TP}{TP+FN} \\
& =\frac {2}{2+4} \\
& =\frac 26=30 \%
\end{align}
$$
​		实际多类别分类任务中，我们通常不满足只通过top-5来衡量一个模型的好坏，而是需要知道从top-1到top-N（N是测试样本的预测类别数，排序后取前N名，只要命中了正确的类别就算作预测正确。）对应的precision和recall。显然随着我们选定的样本越来也多，recall一定会越来越高，而precision整体上会呈下降趋势。把recall当成横坐标，precision当成纵坐标，即可得到常用的precision-recall曲线。这个例子的precision-recall曲线如下：

![img](https://img-blog.csdn.net/20170826194649743?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 计算AP

​		接下来说说AP的计算，此处参考的是PASCAL VOC CHALLENGE的计算方法。

​		首先设定一组阈值，[0, 0.1, 0.2, …, 1]。然后对于recall大于每一个阈值（比如recall>0.3），我们都会得到一个对应的最大precision。这样，我们就计算出了11个precision。AP即为这11个precision的平均值。这种方法英文叫做11-point interpolated average precision。

​		当然PASCAL VOC CHALLENGE自2010年后就换了另一种计算方法。

​		新的计算方法假设这N个样本中有M个正例，那么我们会得到M个recall值（1/M, 2/M, …, M/M）,对于每个recall值r，我们可以计算出对应（r’ > r）的最大precision，然后对这M个precision值取平均即得到最后的AP值。计算方法如下： 

![img](https://img-blog.csdn.net/20170826194741866?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxNDIwMzQ1Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注：这里倒数第二列，top-6的Max Precision应该为3/6（而不是4/7），上面图片有点问题。

## 5.总结

​		AP衡量的是学出来的模型在给定类别上的好坏，而mAP衡量的是学出的模型在所有类别上的好坏，得到AP后mAP的计算就变得很简单了，就是取所有AP的平均值。