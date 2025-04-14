#!/usr/bin/env python3

import json
import math
import threading
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import rospy
from rm_msgs.msg import Plan_State, MoveJ_P


class ArmControl:
    def __init__(self):
        rospy.init_node('d5_arm_control', anonymous=True)

        # 发布手部控制消息
        self.move_hand_pub = rospy.Publisher('/rm_driver/MoveJ_P_Cmd', MoveJ_P, queue_size=10)

        # 加载配置文件
        self.load_config()

    def move_xyz(self, x, y, z):
        def move_hand_callback(msg):
            if msg.state:
                print("手臂移动成功")
            else:
                print("手臂移动失败")
            move_hand_ok_event.set()

        # 订阅是否移动成功
        rospy.Subscriber("/rm_driver/Plan_State", Plan_State, move_hand_callback)
        time.sleep(self.config["subscribe_seconds"])
        move_hand_ok_event = threading.Event()
        

        # 设置坐标
        print(f"设置手臂坐标 ({x}, {y}, {z})")
        hand_point = Point(x, y, z)
        end_point, end_ori_w, end_ori_x, end_ori_y, end_ori_z = self.hand_point_to_end_point(hand_point) 
        moveJ_P_msg = MoveJ_P()
        moveJ_P_msg.Pose.position.x = end_point.x
        moveJ_P_msg.Pose.position.y = end_point.y
        moveJ_P_msg.Pose.position.z = end_point.z
        moveJ_P_msg.Pose.orientation.x = end_ori_x
        moveJ_P_msg.Pose.orientation.y = end_ori_y
        moveJ_P_msg.Pose.orientation.z = end_ori_z
        moveJ_P_msg.Pose.orientation.w = end_ori_w
        moveJ_P_msg.speed = self.config["arm_speed"]
        self.move_hand_pub.publish(moveJ_P_msg)
        print("手臂移动中")

        # 等待设置完成
        move_hand_ok_event.wait()

    def rpy_to_wxyz(self, roll, pitch, yaw):
        # 将角度转换为弧度
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        # 计算四元数
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z

    def hand_point_to_end_point(self, hand_point):
        def sin(deg):
            ret = math.sin(math.radians(deg))
            return ret

        def cos(deg):
            ret = math.cos(math.radians(deg))
            return ret

        def arctan(x, y):
            ret_deg = math.degrees(math.atan2(y, x))
            return ret_deg

        z_offset = self.config["hand_z_offset"]
        downward_angle = self.config["hand_downward_angle"]
        y_offset = self.config["hand_y_offset"]
        z_rotation = -self.config["hand_z_rotation"]

        if downward_angle >= 90 or downward_angle < 0:
            raise ValueError(f"向下角度{downward_angle}应该在0度到90度之间")

        # 位置
        hand_x, hand_y, hand_z = hand_point.get_xyz()
        theta = arctan(hand_x, hand_y)  # 物体与世界坐标系x轴正方向的夹角，[-180°, 180°]
        end_pos_x = hand_x - z_offset * cos(downward_angle) * cos(theta) + y_offset * cos(theta - 90)
        end_pos_y = hand_y - z_offset * cos(downward_angle) * sin(theta) + y_offset * sin(theta - 90)
        end_pos_z = hand_z + z_offset * sin(downward_angle)
        end_point = Point(end_pos_x, end_pos_y, end_pos_z)

        # 姿态，huangting
        end_roll = 0
        end_pitch = -90 - downward_angle
        end_yaw = 180 + theta
        rotation_matrix = R.from_euler('xyz', [end_roll, end_pitch, end_yaw], degrees=True).as_matrix()
        rotate_z_matrix = np.array([
            [cos(z_rotation), sin(z_rotation), 0],
            [-sin(z_rotation), cos(z_rotation), 0],
            [0, 0, 1]
        ])
        end_matrix = rotation_matrix @ rotate_z_matrix
        end_roll, end_pitch, end_yaw = R.from_matrix(end_matrix).as_euler('xyz', degrees=True)
        
        end_ori_w, end_ori_x, end_ori_y, end_ori_z = self.rpy_to_wxyz(end_roll, end_pitch, end_yaw)
        return end_point, end_ori_w, end_ori_x, end_ori_y, end_ori_z
    def load_config(self):
        with open('config.json') as f:
            self.config = json.load(f)


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_xyz(self):
        return self.x, self.y, self.z


class HandParameters:
    def __init__(self, downward_angle, hand_length, z_rotation, y_offset):
        self.downward_angle = downward_angle
        self.hand_length = hand_length
        self.z_rotation = z_rotation
        self.y_offset = y_offset


if __name__ == '__main__':
    armControl = ArmControl()
    x = -0.4
    y = 0
    z = 0.08
    armControl.move_xyz(x, y, z)
    rospy.signal_shutdown("Program finished")  # 显式关闭 ROS 节点