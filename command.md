# RM65-B
## connect
```bash
1. 
# start ROS system
roscore

# change to ROS workspace directory
source ~/ws_rmrobot/devel/setup.bash 
rosrun rm_65_demo 
2. 

source ~/ws_rmrobot/devel/setup.bash && rosrun rm_driver rm_driver 

3. 
rosrun rm_65_demo api_Get_Arm_State_demo
rosrun rm_65_demo main.py


cd ~/ws_rmrobot
source devel/setup.bash 
rosrun rm_65_demo api_moveJ_P_demo

666.
cd ~/ws_rmrobot # change to ROS workspace directory
source devel/setup.bash # load the environment settings for the workspace
roslaunch realsense2_camera rs_camera.launch 

cd ~/ws_rmrobot # change to ROS workspace directory
source devel/setup.bash # load the environment settings for the workspace
roslaunch aruco_ros single.launch


rosrun image_view image_view image:=/aruco_single/result # 显示的图像


rostopic echo /aruco_single/pose # 返回的位姿
```

## compile
```bash
cd ~/ws_rmrobot
catkin build
```

## realsence
```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch realsense2_camera rs_camera.launch # 打开相机节点
rostopic echo /camera/color/camera_info # 查看相机内参
rs-sensor-control
```

# reference
[ROS presentation](https://blog.csdn.net/qq_25267657/article/details/84316111)
[ROS tutorials](http://wiki.ros.org/cn/ROS/Tutorials)
[eye to hand](https://blog.csdn.net/Thinkin9/article/details/123743924)