# RM65-B
```bash

# 初始化ros系统
roscore


# 连接机械臂
source ~/ws_rmrobot/devel/setup.bash && rosrun rm_driver rm_driver 


# 更换ros工作空间
source ~/ws_rmrobot/devel/setup.bash 


# 编译
cd ~/ws_rmrobot && catkin build
```


# 参考资料
[ROS tutorials](http://wiki.ros.org/cn/ROS/Tutorials)
