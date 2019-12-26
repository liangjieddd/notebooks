[TOC]

# Yolov3在VOC训练测试

## 1.准备工作

源代码下载：

~~~
git clone https://github.com/pjreddie/darknet
~~~

进入darknet文件夹下：

~~~
cd darknet
~~~

使用GPU训练，修改Makefile文件

~~~
GPU=1
CUDNN=1
~~~

~~~
#调用摄像头
OPENCV=1
~~~

xubo74118506

