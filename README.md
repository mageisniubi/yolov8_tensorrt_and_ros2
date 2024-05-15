# yolov8_tensorrtx_and_ros2


## 介绍
这是一个将***yolov8***生成的***权重文件.pt***进行tensoret加速的项目，生成***engine***，并结合***ros2***进行消息接收，处理和发布



## 安装教程

打开本地文件夹终端执行`git clone https://gitee.com/mamou_space/yolov5_tensorrt_and_ros2.git`


## 使用说明

```
有n/s/m/l/x/n6/s6/m6/l6/x6这些不同的模式，对于带6的可以输入1280的图片，拥有更高的准确度

```


### 使用与运行 yolov5s 作为例子

1.  从pt生成wts文件

```
//拷贝本工程中的gen_wts.py文件，到yolov5的工程中（训练的工程），并进入
cp {tensorrt}/src/yolov5/src/gen_wts.py {ultralytics}/yolov5
cd {ultralytics}/yolov5
//得到yolov5s.wts文件
python gen_wts.py -w yolov5s.pt -o yolov5s.wts
```

2.  从wts生成engine文件


*  ***修改./src/yolov5/include/yololayer.h中的CLASS_NUM，INPUT_H，INPUT_W***

*  ***修改./src/yolov5/include/names.hpp文件中的数据（一定要大于等于你的数据）***


*  将wts文件拷贝到工作空间中
*  在工作空间中，先编译文件
```
colcon build --packages-select target_bbox_msgs
source install/setup.bash
colcon build
```
*  然后生成engine文件
```
ros2 run yolov5 yolov5 -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // 生成engine文件
ros2 run yolov5 yolov5 -d [.engine] [image folder]  // 使用engine文件
```
*  For example ，samples中是你要测试的图片，大小不能超过你设置的
```

ros2 run yolov5 yolov5 -s yolov5s.wts yolov5s.engine s
ros2 run yolov5 yolov5 -d yolov5s.engine samples

```

3.  结合ros2进行***图像识别***与发布

*  将生成的engine文件拷贝到weights文件夹中

*  修改./src/yolov5/src/yolov5.cpp中 ***NMS_THRESH（置信度）*** 与 ***NMS_THRESH（非极大值抑制）***

*  修改./src/yolov5/launch/yolov5.launch.py 文件中的***parameters***

*  ***注意：处理图像输入应该是BGR模式，如果不是，需要自己修改img_callback，对获取图像进行处理***

***回到工作空间目录下***
```
source install/setup.bash
colcon build
ros2 launch yolov5 yolov5.launch.py

```

4.  使用
    只需要订阅需要的话题即可

## 参与贡献

1.  MYP
