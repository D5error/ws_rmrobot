#!/usr/bin/env python

from util import *
# 长度单位：米
# 角单位：度

camera_matrix = [   
    [607.03796387,   0.    ,     326.52359009],
    [  0.   ,      607.22161865, 244.09521484],
    [  0.    ,       0.      ,     1.        ]
]

extrinsic_matrix = [    
    [-0.51141949,  0.8292543 , -0.22536062, -0.24914425],
    [ 0.79248142 , 0.35371585, -0.49684836 , 0.28937408],
    [-0.33230002, -0.43269204 ,-0.83806581 , 0.51372755],
    [ 0.       ,   0.  ,        0.    ,      1.        ]
]

hand_parameters = Hand_parameters(
    downward_angle=40,
    hand_length=0.09,
    z_rotation=0,
    y_offset=-0.015
)


if __name__ == "__main__":
    rospy.init_node('d5_python_pub', anonymous=True)

    # 外参矩阵和内参矩阵
    # camera_matrix, extrinsic_matrix = get_camera_extrinsic_matrix()

    # # 声音识别
    voice_translate(seconds = 3, model="medium")

    # 获取物体坐标
    boundingbox_u, boundingbox_v, depth = get_boundingbox(8)
    print(f"boundingbox_u = {boundingbox_u}")
    print(f"boundingbox_u = {boundingbox_v}")
    print(f"boundingbox_u = {depth}")

    # 获取物体坐标
    obj_world_x, obj_world_y, obj_world_z = boundingbox_to_world_coordinate(boundingbox_u, boundingbox_v, depth, camera_matrix, extrinsic_matrix)
    print(f"obj_world_x = {obj_world_x}")
    print(f"obj_world_y = {obj_world_y}")
    print(f"obj_world_z = {obj_world_z}")


    move_hand(
        hand_x=obj_world_x,
        hand_y=obj_world_y,
        hand_z=obj_world_z + 0.1,
        speed=0.07,
        hand_parameters=hand_parameters
    )

    grip_control(
        hand_angle=[200, 300, 400, 500, 700, 0],
        hand_speed=100
    )