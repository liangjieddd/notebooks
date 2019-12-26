# resize图片代码

~~~
import os
from PIL import Image

filename = os.listdir(r"D:\deeplearning_workspace\hog_svm_defect_detect\images_postproposal\\")
base_dir = "D:\deeplearning_workspace\hog_svm_defect_detect\images_postproposal\\"
new_dir = "D:\deeplearning_workspace\hog_svm_defect_detect\image_new\\"
size_m = 128
size_n = 128

for img in filename:
    image = Image.open(base_dir + img)
    image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
    image_size.save(new_dir + img)

~~~

