[TOC]

# HOG+SVM训练NEU数据集实现过程记录

## 1.处理数据集

### 1.使用INRIAPerson做一个demo

INRIAPerson数据集结构：

* 70\*134H96

  1.Test->pos:total num:1126张

  ​                          pixel：70\*134



* 96\*160H96

  1.Train->pos:total num:2416

  ​                             pixel:96\*160

* Test

  1. annotation:total num:288

  2. neg:total num:453

     ​		pixel:不定，各种像素都有，大致以320\*640

  ​	3.pos:total num:288

  ​				pixel:不定，都有

   4. annotation.lst:

      ~~~
      Test/annotations/crop_000001.txt
      Test/annotations/crop_000002.txt
      Test/annotations/crop_000003.txt
      Test/annotations/crop_000004.txt
      Test/annotations/crop_000005.txt
      Test/annotations/crop_000006.txt
      Test/annotations/crop_000007.txt
      Test/annotations/crop_000008.txt
      Test/annotations/crop_000009.txt
      Test/annotations/crop_000012.txt
      Test/annotations/crop_000013.txt
      Test/annotations/crop_000015.txt
      Test/annotations/crop_000016.txt
      Test/annotations/crop_000017.txt
      Test/annotations/crop_000018.txt
      Test/annotations/crop_000019.txt
      Test/annotations/crop_000020.txt
      Test/annotations/crop_000021.txt
      Test/annotations/crop_000023.txt
      ~~~

      5.neg.lst:

      ~~~
      Test/neg/00001147.png
      Test/neg/00001148.png
      Test/neg/00001149.png
      Test/neg/00001150.png
      Test/neg/00001153.png
      Test/neg/00001154.png
      Test/neg/00001155.png
      Test/neg/00001156.png
      Test/neg/00001157.png
      Test/neg/00001158.png
      Test/neg/00001159.png
      Test/neg/00001160.png
      Test/neg/00001161.png
      Test/neg/00001162.png
      Test/neg/00001163.png
      ~~~

      6.pos.lst:

      ~~~
      Test/pos/crop_000001.png
      Test/pos/crop_000002.png
      Test/pos/crop_000003.png
      Test/pos/crop_000004.png
      Test/pos/crop_000005.png
      Test/pos/crop_000006.png
      Test/pos/crop_000007.png
      Test/pos/crop_000008.png
      Test/pos/crop_000009.png
      Test/pos/crop_000012.png
      Test/pos/crop_000013.png
      Test/pos/crop_000015.png
      Test/pos/crop_000016.png
      ~~~

* test_64\*128_H96

     1. pos:total num:1126  (图片错误，无法打开)

     2. neg

        ~~~
        ../Test/neg
        ~~~

     3. neg.lst

        ~~~
        test/neg/00001147.png
        test/neg/00001148.png
        test/neg/00001149.png
        test/neg/00001150.png
        test/neg/00001153.png
        test/neg/00001154.png
        test/neg/00001155.png
        test/neg/00001156.png
        test/neg/00001157.png
        test/neg/00001158.png
        test/neg/00001159.png
        test/neg/00001160.png
        ~~~

     4. pos.lst

        ~~~
        test/pos/crop_000001a.png
        test/pos/crop_000001b.png
        test/pos/crop_000002a.png
        test/pos/crop_000002b.png
        test/pos/crop_000003a.png
        test/pos/crop_000003b.png
        test/pos/crop_000004a.png
        test/pos/crop_000004b.png
        test/pos/crop_000005a.png
        ~~~

* Train

  1. annotation：total num：614

     ~~~
     # PASCAL Annotation Version 1.00
     
     Image filename : "Train/pos/crop_000010.png"
     Image size (X x Y x C) : 594 x 720 x 3
     Database : "The INRIA Rh鬾e-Alpes Annotated Person Database"
     Objects with ground truth : 1 { "PASperson" }
     
     # Note that there might be other objects in the image
     # for which ground truth data has not been provided.
     
     # Top left pixel co-ordinates : (0, 0)
     
     # Details for object 1 ("PASperson")
     # Center point -- not available in other PASCAL databases -- refers
     # to person head center
     Original label for object 1 "PASperson" : "UprightPerson"
     Center point on object 1 "PASperson" (X, Y) : (341, 217)
     Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (194, 127) - (413, 647)
     
     ~~~

  2. neg：total num：1218 

     ​			pixel：320\*240 but 各个都有

  3. pos：total num:614 

     ​           pixel:各个像素都有

  4. annotation.lst

     ~~~
     Train/annotations/crop_000010.txt
     Train/annotations/crop_000011.txt
     Train/annotations/crop_000603.txt
     Train/annotations/crop_000606.txt
     Train/annotations/crop_000607.txt
     Train/annotations/crop_000608.txt
     Train/annotations/crop001001.txt
     Train/annotations/crop001002.txt
     Train/annotations/crop001003.txt
     Train/annotations/crop001004.txt
     Train/annotations/crop001005.txt
     ~~~

     

  5. neg.lst

     ~~~
     Train/neg/00000002a.png
     Train/neg/00000003a.png
     Train/neg/00000004a.png
     Train/neg/00000005a.png
     Train/neg/00000006a.png
     Train/neg/00000010a.png
     Train/neg/00000011a.png
     Train/neg/00000012a.png
     Train/neg/00000014a.png
     Train/neg/00000015a.png
     Train/neg/00000030a.png
     Train/neg/00000033a.png
     ~~~

     

  6. pos.lst

     ~~~
     Train/pos/crop_000010.png
     Train/pos/crop_000011.png
     Train/pos/crop_000603.png
     Train/pos/crop_000606.png
     Train/pos/crop_000607.png
     Train/pos/crop_000608.png
     Train/pos/crop001001.png
     Train/pos/crop001002.png
     Train/pos/crop001003.png
     Train/pos/crop001004.png
     Train/pos/crop001005.png
     Train/pos/crop001006.png
     Train/pos/crop001007.png
     ~~~

     

* train_64x128_H96

  1. pos：total num：2416  图片错误

  2. neg

     ~~~
     ../Train/neg
     ~~~

     

  3. neg.lst

     ~~~
     train/neg/00000002a.png
     train/neg/00000003a.png
     train/neg/00000004a.png
     train/neg/00000005a.png
     train/neg/00000006a.png
     train/neg/00000010a.png
     train/neg/00000011a.png
     train/neg/00000012a.png
     train/neg/00000014a.png
     ~~~

     

  4. pos.lst

     ~~~
     train/pos/crop_000010a.png
     train/pos/crop_000010b.png
     train/pos/crop_000011a.png
     train/pos/crop_000011b.png
     train/pos/crop_000603a.png
     train/pos/crop_000603b.png
     train/pos/crop_000606a.png
     train/pos/crop_000606b.png
     train/pos/crop_000606c.png
     ~~~



## 2.HOG提取特征

使用的数据为：

~~~
pos_list = load_images(r'/data/dlj/code/HOG_SVM/INRIAPerson/train_64x128_H96/pos.lst')
    full_neg_lst = load_images(r'/data/dlj/code/HOG_SVM/INRIAPerson/train_64x128_H96/neg.lst')
~~~

