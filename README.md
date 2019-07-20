# Centernet

Simple edition of centernet

代码是根据下面官方code改的，官方的太复杂，所以我改的清晰明了；

如果喜欢请star以下哈

Thanks for the code from https://github.com/Duankaiwen/CenterNet

# 准备环境

1、安装这个conda环境，自己的环境可能会出现很多错误

conda create --name CenterNet --file conda_packagelist.txt

2、 激活这个环境

source activate CenterNet

3、编译几个角点poolling的c文件

cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user

4、编译NMS的c文件

cd <CenterNet dir>/data/coco/PythonAPI
make
  
5、安装coco官方包

cd <CenterNet dir>/data/coco/PythonAPI
make
  
6、按照以下的方式下载和放置图片

Download the training/validation split we use in our paper from here (originally from Faster R-CNN)
Unzip the file and place annotations under <CenterNet dir>/data/coco
Download the images (2014 Train, 2014 Val, 2017 Test) from here
Create 3 directories, trainval2014, minival2014 and testdev2017, under <CenterNet dir>/data/coco/images/
Copy the training/validation/testing images to the corresponding directories according to the annotation files

 




