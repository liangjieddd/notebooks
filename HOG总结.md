[TOC]

# HOG总结

## 方法简介

​		方向梯度直方图（Histogram of Oriented Gradient, HOG）特征是一种在计算机视觉和图像处理中用来进行**物体检测**的描述子。

​		通过计算和统计**局部区域的梯度方向直方图**来构成特征。Hog特征结合SVM分类器已经被广泛应用于图像识别中，尤其在行人检测中获得了极大的成功。现如今如今虽然有很多行人检测算法不断提出，但基本都是以HOG+SVM的思路为主。

### 主要思想

​		在一幅图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。

​		其本质是**梯度的统计信息**，而梯度主要存在于边缘所在的地方。

### 实现过程

​		简单来说，首先需要将图像分成小的连通区域，称之为细胞单元。然后采集细胞单元中各像素点的梯度或边缘的方向直方图。最后把这些直方图组合起来就可以构成特征描述器。

### 算法优点

​		与其他的特征描述方法相比，HOG有较多优点。

1. 由于HOG是在图像的局部方格单元上操作，所以它对图像几何的和光学的形变都能保持很好的**不变性**，这两种形变只会出现在更大的空间领域上。
2. 其次，在粗的空域抽样、精细的方向抽样以及较强的局部光学归一化等条件下，只要行人大体上能够保持直立的姿势，可以容许行人有一些细微的肢体动作，这些细微的动作可以被忽略而不影响检测效果。
3. 因此HOG特征是特别适合于做图像中的人体检测的。

## HOG流程

HOG特征提取算法的整个实现过程大致如下：

1. 读入所需要的检测目标即输入的image
2. 将图像进行灰度化（将输入的彩色的图像的r,g,b值通过特定公式转换为灰度值）
3. 采用Gamma校正法对输入图像进行颜色空间的标准化（归一化）
4. 计算图像每个像素的梯度（包括大小和方向），捕获轮廓信息
5. 统计每个cell的梯度直方图（不同梯度的个数），形成每个cell的descriptor
6. 将每几个cell组成一个block（以3*3为例），一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
7. 将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，这就是最终分类的特征向量

HOG参数设置是：2\*2 cell／block、8\*8 pixel／cell、8个直方图通道、步长为1。

特征提取流程：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502105643118?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## Python实现

### 1.数据准备

​		读入彩色图像，并转换为灰度值图像, 获得图像的宽和高。

​		采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目的是调节图像的对比度，降低图像局部的阴影和光照变化所造成的影响，同时可以抑制噪音。

​		采用的gamma值为0.5。

~~~
#first part

import cv2
import numpy as np

img = cv2.imread('person_037.png',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Image', img)
# cv2.imwrite("Image-test.jpg", img)
# cv2.waitKey(0)

img = np.sqrt(img / float(np.max(img)))
# cv2.imshow('Image', img)
# cv2.imwrite("Image-test2.jpg", img)
# cv2.waitKey(0)
~~~

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502115222192?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502123153854?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 计算每个像素的梯度

​		计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；

​		**求导操作不仅能够捕获轮廓，人影和一些纹理信息，还能进一步弱化光照的影响。**

​		求出输入图像中像素点（x,y）处的**水平方向梯度**、**垂直方向梯度**和**像素值**，从而求出**梯度幅值和方向**。

> 常用的方法是：首先用[-1,0,1]梯度算子对原图像做卷积运算，得到  x方向（水平方向，以向右为正方向）的梯度分量gradscalx，
>
> ​		然后用[1,0,-1]T梯度算子对原图像做卷积运算，得到  y方向（竖直方向，以向上为正方向）的梯度分量 gradscaly。然后再用以上公式计算该像素点的梯度大小和方向。

~~~
# second part

height, width = img.shape

gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)

gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)

print gradient_magnitude.shape, gradient_angle.shape

~~~

out:

~~~
(640, 480) (640, 480)
~~~



### 为每个细胞单元构建梯度方向直方图

​		我们将图像分成若干个“单元格cell”，默认我们将cell设为8\*8个像素。

​		假设我们采用8个bin的直方图来统计这6*6个像素的梯度信息。也就是将cell的梯度方向360度分成8个方向块，例如：如果这个像素的梯度方向是0-22.5度，直方图第1个bin的计数就加一，这样，对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），就可以得到这个cell的梯度方向直方图了，就是该cell对应的8维特征向量而梯度大小作为投影的权值。

~~~
# third part

cell_size = 8
bin_size = 8
angle_unit = 360 / bin_size

gradient_magnitude = abs(gradient_magnitude)

cell_gradient_vector = np.zeros((height / cell_size, width / cell_size, bin_size))

print cell_gradient_vector.shape

def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit)%8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size]
        print cell_angle.max()

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

~~~



### 可视化Cell梯度直方图

将得到的每个cell的梯度方向直方图绘出，得到特征图：

~~~
# fourth part

import math
import matplotlib.pyplot as plt

hog_image= np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()

for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag
        angle = 0
        angle_gap = angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap

plt.imshow(hog_image, cmap=plt.cm.gray)
plt.show()

~~~

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502114650793?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 统计Block的梯度信息

​		把细胞单元组合成大的块(block），块内归一化梯度直方图；

​		由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大。这就需要对梯度强度做归一化。**归一化能够进一步地对光照、阴影和边缘进行压缩。**

​		把各个细胞单元组合成大的、空间上连通的区间（blocks）。这样，一个block内所有cell的特征向量串联起来便得到该block的HOG特征。这些区间是互有重叠的，

​		本次实验采用的是矩阵形区间，它可以有三个参数来表征：每个区间中细胞单元的数目、每个细胞单元中像素点的数目、每个细胞的直方图通道数目。

​		本次实验中我们采用的参数设置是：2\*2细胞／区间、8\*8像素／细胞、8个直方图通道,步长为1。则一块的特征数为2\*2\*8。

~~~
# fifth part

hog_vector = []

for i in range(cell_gradient_vector.shape[0] - 1):
    for j in range(cell_gradient_vector.shape[1] - 1):
        block_vector = []
        block_vector.extend(cell_gradient_vector[i][j])
        block_vector.extend(cell_gradient_vector[i][j + 1])
        block_vector.extend(cell_gradient_vector[i + 1][j])
        block_vector.extend(cell_gradient_vector[i + 1][j + 1])
        mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
        
        magnitude = mag(block_vector)
        if magnitude != 0:
            normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
            block_vector = normalize(block_vector, magnitude)
        hog_vector.append(block_vector)

print np.array(hog_vector).shape

~~~

out

~~~
(4661, 32)
~~~

共有4661个block，每个block有32维的特征

## 代码封装

~~~
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((height / self.cell_size, width / self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

img = cv2.imread('person_037.png', cv2.IMREAD_GRAYSCALE)
hog = Hog_descriptor(img, cell_size=8, bin_size=8)
vector, image = hog.extract()
print np.array(vector).shape
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
~~~

## 结果分析

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502123413923?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​		能够更加有效的区分梯度显示边缘。这是因为对各个像素的梯度进行了**全局归一化**，并且在描绘梯度方向时加入了**梯度量级的非线性映射**，使得梯度方向产生明显的深浅和长度差异，更易于区分边缘，凸显明显的梯度变化。

​		此外在输入图像时，采用Gamma校正对输入图像进行颜色空间的标准化能够抑制噪声，使得产生的边缘更加明显，清晰。

​		此外改变cell的大小和直方图方向通道的效果如下： 
​		cell_size = 10 即 16*16个像素

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502123459361?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​		可以看出增大cell的size得到的特征图更加注重基本轮廓和边缘，而忽略一些细节，某种程度上降低了噪声。

​		当通道数目为16个方向

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170502123530058?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcHBwODMwMDg4NQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

梯度特征图像的细节变得更加明显，方向更多。