#!/usr/bin/env python

from util import *
# 长度单位：米
# 角单位：度


if __name__ == "__main__":
    # 外参矩阵和内参矩阵
    camera_matrix, extrinsic_matrix = get_camera_extrinsic_matrix()

    # 声音识别
    voice_translate(seconds = 2, model="small")

    # 获取物体坐标
    boundingbox_u, boundingbox_v, depth = get_boundingbox()

    # 获取物体坐标
    obj_world_x, obj_world_y, obj_world_z = get_world_coords_from_boundingbox(boundingbox_u, boundingbox_v, depth, camera_matrix, extrinsic_matrix)

    # print(f"obj_world_x = {obj_world_x}")
    # print(f"obj_world_y = {obj_world_y}")
    # print(f"obj_world_z = {obj_world_z}")
    # # exit()
    # obj_world_x, obj_world_y, obj_world_z = -0.4, -0.4, 0

    end_position_x, end_position_y, end_position_z, end_roll, end_pitch, end_yaw = get_end_from_obj(
        obj_x = obj_world_x,
        obj_y = obj_world_y,
        obj_z = obj_world_z,
        beta = 40,
        hand_length = 0.2,
        offset = 0.03,
        height = 0.13,

        hand_z_rotation = 30
    )

    end_orientation_w, end_orientation_x, end_orientation_y, end_orientation_z = get_wxyz_from_rpy(end_roll, end_pitch, end_yaw)

    speed = 0.07
    publish_to_ros(end_position_x, end_position_y, end_position_z, end_orientation_x, end_orientation_y, end_orientation_z, end_orientation_w, speed)
