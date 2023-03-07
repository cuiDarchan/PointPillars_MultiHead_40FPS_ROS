# Pointpillars_ROS工程化
**工程参考：**  
[1. PointPillars_MultiHead_40FPS](https://github.com/hova88/PointPillars_MultiHead_40FPS)
[2. PointPillars 工程复现](https://blog.csdn.net/weixin_36354875/article/details/126051498?spm=1001.2014.3001.5502)
[3. DeepDriving -- 激光点云3D目标检测算法之PointPillars](https://blog.csdn.net/weixin_44613415/article/details/125800169)
---

## 效果及简单说明
本工程主要基于[PointPillars_MultiHead_40FPS](https://github.com/hova88/PointPillars_MultiHead_40FPS)，在其基础上添加`ROS节点`，实现了实时展示 。
**1) Pointpillars ROS节点动态展示**
<p align="left">
  <img width="2000" alt="pp" src=docs/pointpillars.gif>
</p>

**2）左图为本仓库实现的demo，右图为OpenPCdet实现的demo**
<p align="left">
  <img width="2000" alt="fig_method" src=docs/demo.png>
</p>

---

## 使用步骤
## 1. 生成模型文件   
**详细参考：**
[1. PointPillars 工程复现](https://blog.csdn.net/weixin_36354875/article/details/126051498?spm=1001.2014.3001.5502)

## 2. 编译执行  
```
mkdir build && cd build
cmake .. && make -j8
./pointpillars_ros #执行ros节点
rviz                                    # 可以打开工程目录下rviz下的配置，播放nuscenes的bag
```
## 3. 配置文件修改  
`bootstrap.yaml`重要参数：   
```
LidarTopic: lidar_top                   # 订阅lidar消息
BoundingBoxTopic: boundingbox_objects   # 障碍物消息，障碍物显示需要插件
ObjectScoreThreshold: 0.3               # 障碍物打分阈值
```
 ## 4. 结尾  
 图中可见初步效果，可以修改打分阈值来改善一部分，后期添加`障碍物追踪`可以更稳定一些。