#!/usr/bin/env python

from util import *
# 长度单位：米
# 角单位：度


if __name__ == "__main__":
    rospy.init_node('d5_python_pub', anonymous=True)

    # 外参矩阵和内参矩阵
    # camera_matrix, extrinsic_matrix = get_camera_extrinsic_matrix()

    camera_matrix = [[607.03796387,   0.    ,     326.52359009],
 [  0.   ,      607.22161865, 244.09521484],
 [  0.    ,       0.      ,     1.        ]]
    
    extrinsic_matrix = [[-0.51141949,  0.8292543 , -0.22536062, -0.24914425],
 [ 0.79248142 , 0.35371585, -0.49684836 , 0.28937408],
 [-0.33230002, -0.43269204 ,-0.83806581 , 0.51372755],
 [ 0.       ,   0.  ,        0.    ,      1.        ]]

    # # 声音识别
    voice_translate(seconds = 3, model="medium")

    # 获取物体坐标
    file_path = r"./ii7_out_bbox.txt"  # 替换为你的文件路径
    if os.path.exists(file_path):
        os.remove(file_path)
    boundingbox_u, boundingbox_v, depth = get_boundingbox(8)

    print(f"boundingbox_u = {boundingbox_u}")
    print(f"boundingbox_u = {boundingbox_v}")
    print(f"boundingbox_u = {depth}")

    # 获取物体坐标
    obj_world_x, obj_world_y, obj_world_z = get_world_coords_from_boundingbox(boundingbox_u, boundingbox_v, depth, camera_matrix, extrinsic_matrix)

    print(f"obj_world_x = {obj_world_x}")
    print(f"obj_world_y = {obj_world_y}")
    print(f"obj_world_z = {obj_world_z}")




    obj_world_x = -0.4
    obj_world_y = 0.15
    obj_world_z = 0

    end_position_x, end_position_y, end_position_z, end_roll, end_pitch, end_yaw = get_end_from_obj(
        obj_x = obj_world_x,
        obj_y = obj_world_y,
        obj_z = obj_world_z,
        beta = 40,
        hand_length = 0.11,
        offset = 0.015,
        height = 0.09,
        hand_z_rotation = 30
    )

    end_orientation_w, end_orientation_x, end_orientation_y, end_orientation_z = get_wxyz_from_rpy(end_roll, end_pitch, end_yaw)

    speed = 0.07
    publish_to_ros(end_position_x, end_position_y, end_position_z, end_orientation_x, end_orientation_y, end_orientation_z, end_orientation_w, speed)

    hand_grip(
        hand_angle = [200, 300, 400, 500, 700, 0], 
        hand_speed = 100,
        ori_w=end_orientation_w, 
        ori_x=end_orientation_x,
         ori_y= end_orientation_y,
          ori_z= end_orientation_z
    )