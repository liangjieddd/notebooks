[TOC]

# 从SNE到t-SNE再到LargeVis

## 1.前言

​		数据可视化是大数据领域非常倚重的一项技术，但由于业内浮躁的大环境影响，这项技术的地位渐渐有些尴尬。尤其是在诸如态势感知、威胁情报等应用中，简陋的可视化效果太丑，过于华丽的可视化效果只能忽悠忽悠外行，而给内行的感觉就是刻意为之、华而不实。

​		曾几何时，可视化技术不过是一种数据分析的手段罢了。惭愧的说就是我们的算法还不够智能，必须依靠人类的智慧介入分析。所以，需要通过可视化技术把高维空间中的数据以二维或三维的形式展示给我们这样的低维生物看，展示的效果如何也就直接决定着我们分析的难度。

​		抛开浮躁的大环境，在数据可视化领域还是有人踏踏实实做研究的，比如深度学习大牛Hinton(SNE)、Maaten(t-SNE)还有[唐建大神](http://research.microsoft.com/en-us/people/jiatang/)(LargeVis，新鲜出炉，*WWW’16*最佳论文提名)，下面言归正传，我们从简单的基础知识开始。

## 2.预备知识

### 2.1 降维

​		降维顾名思义就是把数据或特征的维数降低，一般分为线性降维和非线性降维，比较典型的如下：

- 线性降维

  - PCA(Principal Components Analysis)
  - LDA(Linear Discriminant Analysis)
  - MDS(Classical Multidimensional Scaling)

- 非线性降维：Isomap(Isometric Mapping)

  - LLE(Locally Linear Embedding)
  - LE(Laplacian Eigenmaps) 

  ​        大家可能对线性降维中的一些方法比较熟悉了，但是对非线性降维并不了解，非线性降维中用到的方法大多属于流形学习方法。

### 2.1 流形学习

​		流形学习(Manifold Learning)听名字就觉得非常深奥，涉及微分流行和黎曼几何等数学知识。当然，想要了解流形学习并不需要我们一行一行的去推导公式，通过简单的例子也能够有一个直观的认识。关于流行学习的科普文章首推pluskid写的[《浅谈流行学习》](http://blog.pluskid.org/?p=533)，里面有很多通俗易懂的例子和解释。

​		简单来说，地球表面就是一个典型的流形，在流形上计算距离与欧式空间有所区别。例如，计算南极与北极点之间的距离不是从地心穿一个洞计算直线距离，而是沿着地球表面寻找一条最短路径，这样的一条路径称为**测地线**。如下面所示的三幅图

![流形与测地线](http://lc-cf2bfs1v.cn-n1.lcfile.com/6a0a667a8081616f.png)

​		其中第一张图为原始数据点分布，红色虚线是欧式距离，蓝色实线是沿着流形的真实测地线距离。

​		第二张图是在原始数据点的基础上基于欧式距离构造的kNN图（灰色线条，下面还会具体介绍kNN图），红色实线表示kNN图中两点之间的最短路径距离。

​		第三张图是将流形展开后的效果，可以看到，kNN图中的最短路径距离（红色实线）要略长于真实测地线距离（蓝色实线）。

​		在实际应用中，真实测地距离较难获得，一般可以通过构造kNN图，在kNN图中寻找最短路径距离作为真实测地线距离的近似。

### 2.3 t分布

​		大家在概率与统计课程中都接触过t分布的概念，从正态总体中抽取容量为N的随机样本，若该正态总体的均值为μ，方差为
$$
{\sigma}^2
$$
。随机样本均值为x¯，方差为
$$
s^2=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})^2
$$
，随机变量t可表示为：
$$
t=\frac{\bar{x}-\mu}{s/\sqrt{N}}
$$


此时我们称t服从自由度为n−1的t分布，即t∼t(n−1)

​		下图展示了不同自由度下的t分布形状与正态分布对比，其中自由度为1的t分布也称为柯西分布。

​		**自由度越大，t分布的形状越接近正态分布。**

![正态分布与t分布](http://lc-cf2bfs1v.cn-n1.lcfile.com/823840b0a7738efb.png)

​		从图中还可以看出，t分布比正态分布要“胖”一些，尤其在尾部两端较为平缓。t分布是一种典型的长尾分布。实际上，在[稳定分布](https://en.wikipedia.org/wiki/Stable_distribution)家族中，**除了正态分布，其他均为长尾分布**。长尾分布有什么好处呢？在处理小样本和一些异常点的时候作用就突显出来了。下文介绍t-sne算法时也会涉及到t分布的长尾特性。

### 2.4 kNN图

​		kNN图(k-Nearest Neighbour Graph)实际上是在经典的kNN(k-Nearest Neighbor)算法上增加了一步构图过程。

​		假设空间中有n个节点，对节点vi，通过某种距离度量方式（欧式距离、编辑距离）找出距离它最近的k个邻居v1,v2,⋯,vk，然后分别将vi与这k个邻居连接起来，形成k条有向边。对空间中所有顶点均按此方式进行，最后就得到了kNN图。

当然，为方便起见，在许多场景中我们往往将kNN图中的有向边视为无向边处理。如下图是一个二维空间中以欧式距离为度量的kNN图。

![kNN图样例](http://lc-cf2bfs1v.cn-n1.lcfile.com/94568c30f8b7698c.png)

​		kNN图的一种用途上文已经提到过：在计算流形上的测地线距离时，可以构造基于欧式距离的kNN图得到一个近似。原因很简单，我们可以把一个流形在很小的局部邻域上近似看成欧式的，也就是局部线性的。这一点很好理解，比如我们所处的地球表面就是一个流形，在范围较小的日常生活中依然可以使用欧式几何。但是在航海、航空等范围较大的实际问题中，再使用欧式几何就不合适了，使用黎曼几何更加精确。

​		**kNN图还可用于异常点检测。**在大量高维数据点中，一般正常的数据点会聚集为一个个簇，而异常数据点与正常数据点簇的距离较远。通过构建kNN图，可以快速找出这样的异常点。

### 2.5 k-d树与随机投影树

​		刚才说到kNN图在寻找流形的过程中非常有用，那么如何来构建一个kNN图呢？

​		常见的方法一般有三类：

* 第一类是空间分割树(space-partitioning trees)算法

* 第二类是局部敏感哈希(locality sensitive hashing)算法

* 第三类是邻居搜索(neighbor exploring techniques)算法

  其中k-d树和随机投影树均属于第一类算法。



​		很多同学可能不太熟悉随机投影树(Random Projection Tree)，但一般都听说过k-d树。

​		k-d树是一种**分割k维数据空间**的数据结构，本质上是一棵**二叉树**。主要用于**多维空间关键数据的搜索**，如范围搜索、最近邻搜索等。那么如何使用k-d树搜索k近邻，进而构建kNN图呢？我们以二维空间为例进行说明，如下图所示：

![k-d树示意图](http://lc-cf2bfs1v.cn-n1.lcfile.com/1bd1eadb508a9986.png)

​		上图是一个二维空间的k-d树，构建k-d树是一个**递归**的过程，根节点对应区域内所有点，将空间按某一维划分为左子树和右子树之后，重复根结点的分割过程即可得到下一级子节点，直到k-d树中所有叶子节点对应的点个数小于某个阈值。

​		有了k-d树之后，我们寻找k近邻就不用挨个计算某个点与其他所有点之间的距离了。例如寻找下图中红点的k近邻，只需要搜索当前子空间，同时不断回溯搜索父节点的其他子空间，即可找到k近邻点。

![k-d树找k近邻](http://lc-cf2bfs1v.cn-n1.lcfile.com/f41ba87492c96da2.png)

​		当然，搜索过程还有一些缩小搜索范围的方法，例如画圆判断是否与父节点的分割超平面相交等等，这里就不展开讨论了。

​		不过k-d树最大的问题在于其划分空间的方式比较死板，是严格按照坐标轴来的。对高维数据来说，就是将高维数据的每一维作为一个坐标轴。当数据维数较高时，k-d树的深度可想而知，**维数灾难**问题也不可避免。

​		相比之下，**随机投影树**划分空间的方式就比较灵活，还是以二维空间为例，如下图所示：

![随机投影树示意图](http://lc-cf2bfs1v.cn-n1.lcfile.com/459e3b29b5065043.png)

​		随机投影树的基本思路还是与k-d树类似的，不过划分空间的方式不是按坐标轴了，而是按**随机产生的单位向量**。有的同学说，这样就能保证随机投影树的深度不至于太深吗？随机产生的单位向量有那么靠谱吗？这里需要注意的是，我们所分析的数据处于一个流形上的，并非是杂乱无章的，因此从理论上讲，随机投影树的深度并不由数据的维数决定，而取决于数据所处的流形维数。(此处可参考Freund等人的论文《Learning the structure of manifolds using random projections》)

​		那么如何使用随机投影树寻找k近邻呢？当然可以采用和k-d树类似回溯搜索方法。但是当我们对k近邻的精确度要求不高时，可以采用一个更加简单巧妙的方式，充分利用随机投影树的特性。

​		简单来说，我们可以**并行的构建多个随机投影树**，由于划分的单位向量都是随机产生的，因此每棵随机投影树对当前空间的划分都是不相同的，如下图所示

![随机投影树找k近邻](http://lc-cf2bfs1v.cn-n1.lcfile.com/644b8aaa16d0b2a5.png)

​		例如我们想搜索红点的k近邻，只需要在不同的随机投影树中搜索其所处的子空间(或者仅回溯一层父结点)，最后取并集即可。这样做虽然在构建随机投影树的过程中较为耗时耗空间，但是在搜索阶段无疑是非常高效的。

### 2.6 LINE

​		LINE，即Large-scale Information Network Embedding，是唐建大神2015年的一项工作(*www’15*)。内容依旧很好很强大，而且代码是开源的。

​		一句话概括，LINE是“Embed Everything”思想在网络表示中的发扬光大。自从Mikolov开源word2vec以来，词向量(word embedding)的概念在NLP界可谓是火的一塌糊涂，embedding的概念更是快速渗透到其他各研究领域。entity embedding、relation embedding…等如雨后春笋般涌现，更是有人在Twitter上犀利的吐槽：

![twitter吐槽](http://lc-cf2bfs1v.cn-n1.lcfile.com/87a9088866752f77.png)

​		当然，这里完全没有贬低LINE的意思，事实上LINE的工作是非常出色的，主要有两大突出贡献：

​		一是能够适应各种类型(无向边或有向边、带权值不带权值的)的大规模(百万级节点、十亿级边)网络，而且能够很好的捕获网络中的一阶和二阶相似性；

​		二是提出了非常给力的**边采样算法**(edge-sampling algorithm)，大幅降低了LINE的时间复杂度，使用边采样算法后时间复杂度与网络中边的数量呈线性关系。LargeVis的高效也得益于LINE及其边采样算法。

​		一阶相似性指的是网络中**两个节点**之间的点对相似性，具体为**节点之间边的权重**(如果点对不存在边，则其一阶相似性为0)；

​		二阶相似性指的是**若节点间共享相似的邻居节点，那么两者就趋于相似**。比如下图展示的这种情况，边的权值大小用粗细表示：

![一阶二阶相似度](http://lc-cf2bfs1v.cn-n1.lcfile.com/f233ea2102c812f3.png)

​		其中节点8与节点9之间的一阶相似性为较高，因为其直接连接边的权值较高。节点1与节点7有着绝大多数相同的邻居，因此两者的二阶相似性非常高。

​		边采样算法的思路来源于Mikolov在word2vec中使用的**负采样**优化技术。既提高了训练的效率，也解决了网络表示中带权值边在训练过程中造成的梯度剧增问题，具体的边采样算法在下文涉及的地方进行介绍。

### 2.7 负采样

​		了解word2vec的同学一定对负采样(Negative sampling)不陌生，Mikolov在word2vec中集成了CBOW和Skip-gram两种词向量模型，在训练过程中使用到了多项优化技术，**负采样正是其中一种优化技术**。

​		我们以Skip-gram模型为例进行说明，Skip-gram模型的思路是**从目标词预测上下文，用一个上下文窗口限定文本范围**，如下图所示：

![Skip-gram](http://lc-cf2bfs1v.cn-n1.lcfile.com/f6397e4873f8c449.png)

​		Skip-gram模型需要最大化“做了”“一点”“的”“贡献”等词语出现在目标词“微小”周围的概率，即最大化p(c∣w)=∑p(wi∣w)。

​		出现在目标词周围上下文窗口中的词wi∈c构成一个正样本(wi,w)，未出现在目标词周围的词wj∈D构成负样本(wj,w)。

​		我们在训练过程中要**最大化正样本出现的概率**，同时也要**减小负样本出现的概率**。为什么要减小负样本出现的概率呢，只提高正样本出现的概率不就可以了吗？举个不恰当的例子，这就好比垄断一样，为了达到最终目的不择手段，一方面肯定要加强自身产品的竞争力，另一方面竞争对手也在发展壮大，发展水平并不比我们差，所以必须使用些手段，打压消灭竞争对手，负采样也是这么个道理。

​		由于负样本数量众多(上下文窗口之外的词基本都可以构成负样本)，直接考虑所有的负样本显然是不现实的，所以我们用采样的方式选一部分负样本出来即可。那么负采样具体如何采样呢？在语料中有的词语出现频率高，有的词语出现频率低，直接从词表中随机抽取负样本显然是不科学的。

​		word2vec中使用的是一种**带权采样**策略，即根据词频进行采样，高频词被采样的概率较大，低频词被采样的概率较小。

​		那么具体如何带权采样呢？看下面这张图，词wi的词频用fwi表示

![带权采样](http://lc-cf2bfs1v.cn-n1.lcfile.com/cd91f11ea9c8d27f.png)

​		上面那根线段是按词频进行分割的，词频越高线段较长，下面的线段是等距离分割。我们往下方的线段中随机打点(均匀分布)，根据点所落在的区间对应到上方的线段，即可确定所采样的词。直观来看，采用这种方式词频较高的词被采样到的概率更大，词频较低的词被采样到的概率更低。

加入负采样优化之后，目标函数的形式变为

$$
\log \sigma (v_{w_c}^T \cdot v_w)+\sum_{i=1}^k\ _{E_{w_i}\sim P_n(f)}[\log \sigma(-v_{w_i}^T\cdot v_w)]
$$
其中w表示目标词，wc表示目标词周围上下文窗口中的词(正样本)，wi表示未出现在上下文窗口中的词(负样本)，k表示抽取的负样本个数，Pn(f)是用于负样本生成的噪声分布，f表示词频，
$$
P_n(f)\propto f^{0.75}
$$
，不要问我0.75怎么来的，Mikolov做实验得出来的，直接用原始词频效果不好，加个0.75次幂效果较好。

word2vec里面还有许多有意思的细节，感兴趣的同学可以去看看peghoty写的《[word2vec中的数学原理](http://blog.csdn.net/itplus/article/details/37998797)》

## 3.从SNE说起

​		了解完预备知识后，我们可以从SNE开始本趟可视化算法之旅了。

​		SNE即stochastic neighbor embedding，是Hinton老人家2002年提出来的一个算法，出发点很简单：**在高维空间相似的数据点，映射到低维空间距离也是相似的**。

​		常规的做法是用欧式距离表示这种相似性，而SNE把这种距离关系转换为一种**条件概率**来表示相似性。

​		什么意思呢？考虑高维空间中的两个数据点xi和xj，xi以条件概率pj∣i选择xj作为它的邻近点。考虑以xi为中心点的高斯分布，若xj越靠近xi，则pj∣i越大。反之，若两者相距较远，则pj∣i极小。因此，我们可以这样定义pj∣i：
$$
p_{j|i}=\frac{\exp (-\left \| x_i-x_j \right \|^2/2 \sigma_{i}^2)}{\sum_{k \neq i}\exp (-\left \| x_i-x_k \right \|^2/2 \sigma_{i}^2)}
$$
其中σi表示以xi为中心点的高斯分布的方差。由于我们只关心不同点对之间的相似度，所以设定pi∣i=0。

​		当我们把数据映射到低维空间后，高维数据点之间的相似性也应该在低维空间的数据点上体现出来。这里同样用条件概率的形式描述，假设高维数据点xi和xj在低维空间的映射点分别为yi和yj。类似的，低维空间中的条件概率用qj∣i表示，并将所有高斯分布的方差均设定为1/√2，所以有：
$$
q_{j|i}=\frac{\exp (-\left \| y_i-y_j \right \|^2)}{\sum_{k \neq i}\exp (-\left \| y_i-y_k \right \|^2)}
$$
同理，设定qi∣i=0。此时就很明朗了，若yi和yj真实反映了高维数据点xi和xj之间的关系，那么条件概率pj∣i与qj∣i应该完全相等。这里我们只考虑了xi与xj之间的条件概率，若考虑xi与其他所有点之间的条件概率，则可构成一个条件概率分布Pi，同理在低维空间存在一个条件概率分布Qi且应该与Pi一致。如何衡量两个分布之间的相似性？当然是用经典的KL距离(Kullback-Leibler Divergence)，SNE最终目标就是对所有数据点最小化这个KL距离，我们可以使用梯度下降算法最小化如下代价函数：
$$
C=\sum_{i}KL(P_i||Q_i)=\sum_i \sum_j p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}
$$
似乎到这里问题就漂亮的解决了，你看我们代价函数都写出来了，剩下的事情就是利用梯度下降算法进行训练了。

​		但事情远没有那么简单，因为KL距离是一个非对称的度量。最小化代价函数的目的是让pj∣i和qj∣i的值尽可能的接近，即**低维空间中点的相似性应当与高维空间中点的相似性一致**。但是从代价函数的形式就可以看出，当pj∣i较大，qj∣i较小时，代价较高；而pj∣i较小，qj∣i较大时，代价较低。

​		什么意思呢？很显然，高维空间中两个数据点距离较近时，若映射到低维空间后距离较远，那么将得到一个很高的惩罚，这当然没问题。反之，高维空间中两个数据点距离较远时，若映射到低维空间距离较近，将得到一个很低的惩罚值，这就有问题了，理应得到一个较高的惩罚才对。**换句话说，SNE的代价函数更关注局部结构，而忽视了全局结构。**

SNE代价函数对yi求梯度后的形式如下：
$$
\frac{\delta C}{\delta y_i}=2\sum_{j}(p_{j|i}-q_{j|i}+p_{i|j}-q_{i|j})(y_i-y_j)
$$
​		这个梯度还有一定的物理意义，我们可以用分子之间的引力和斥力进行解释。

​		低维空间中点yi的位置是由其他所有点对其作用力的合力所决定的。其中某个点yj对其作用力是沿着yi−yj方向的，具体是引力还是斥力占主导就取决于yj与yi之间的距离了，其实就与
$$
(p_{j\mid i}-q_{j\mid i}+p_{i\mid j}-q_{i\mid j})
$$
这一项有关。

​		SNE算法中还有一个细节是关于高维空间中以点xi为中心的正态分布方差σi的选取，这里不展开讲了，有兴趣的同学可以去看看论文。

​		最后，我们来看一下SNE算法的效果图。将SNE算法用在UPS database的手写数字数据集上(五种数字，01234)，效果如下：

![sne效果图](http://lc-cf2bfs1v.cn-n1.lcfile.com/5eddecaad72424c3.png)

​		从图中可以看出，SNE的可视化效果还算可以，同一类别的数据点映射到二维空间后基本都能聚集在一起，但是不同簇之间的边界过于模糊。老实说，如果不是这个图上把不同类别用不同颜色和符号标识出来，根本没法把边界处的数据点区分开来，做可视化分析也非常不方便。这个问题下面我们还会详细分析。

[http://bindog.github.io/blog/2016/06/04/from-sne-to-tsne-to-largevis/#0x05-%E4%BB%8Et-sne%E5%86%8D%E5%88%B0largevis%E5%8E%9A%E7%A7%AF%E8%96%84%E5%8F%91](http://bindog.github.io/blog/2016/06/04/from-sne-to-tsne-to-largevis/#0x05-从t-sne再到largevis厚积薄发)

