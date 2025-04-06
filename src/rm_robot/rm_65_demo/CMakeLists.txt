cmake_minimum_required(VERSION 3.0.2)
project(rm_65_demo)

find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  moveit_core
  moveit_visual_tools
  moveit_ros_planning
  moveit_ros_planning_interface
  pluginlib
  geometric_shapes
  pcl_ros
  pcl_conversions
  rosbag
  tf
  tf2_ros
  tf2_geometry_msgs
  rm_msgs
  cv_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rm_65_demo
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
include 
 ${catkin_INCLUDE_DIRS}
 ${Sophus_INCLUDE_DIRS}
 ${PROJECT_SOURCE_DIR}/include
)



add_executable(planning_scene_ros_api_demo src/planning_scene_ros_api_demo.cpp)
add_executable(pick_place_demo src/pick_place_demo.cpp)
add_executable(test_api_movel src/test_api_movel.cpp)
add_executable(api_moveJ_demo src/api_moveJ_demo.cpp)
add_executable(api_moveL_demo src/api_moveL_demo.cpp)
add_executable(api_moveJ_P_demo src/api_moveJ_P_demo.cpp)
add_executable(api_ChangeToolName_demo src/api_ChangeToolName_demo.cpp)
add_executable(api_ChangeWorkFrame_demo src/api_ChangeWorkFrame_demo.cpp)
add_executable(api_Get_Arm_State_demo src/api_Get_Arm_State_demo.cpp)
add_executable(api_teach_demo src/api_teach_demo.cpp)
add_executable(getArmState_publisher src/api_getArmCurrentState.cpp)



target_link_libraries(planning_scene_ros_api_demo ${catkin_LIBRARIES})
target_link_libraries(pick_place_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(test_api_movel ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_moveJ_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_moveL_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_moveJ_P_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_ChangeToolName_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_ChangeWorkFrame_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_Get_Arm_State_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(api_teach_demo ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(getArmState_publisher ${catkin_LIBRARIES} ${Boost_LIBRARIES})


catkin_install_python(PROGRAMS scripts/hand.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)