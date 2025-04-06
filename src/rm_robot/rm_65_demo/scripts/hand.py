#!/usr/bin/env python3
import json
import threading
import time

import rospy
from std_msgs.msg import Bool
from rm_msgs.msg import Hand_Angle, Hand_Speed


class Hand:
    def __init__(self):
        rospy.init_node('d5_hand', anonymous=True)

        # 灵巧手角度发布
        self.hand_angle_pub = rospy.Publisher("/rm_driver/Hand_SetAngle",Hand_Angle, queue_size=10)

        # 灵巧手速度发布
        self.hand_speed_pub = rospy.Publisher("/rm_driver/Hand_SpeedAngle", Hand_Speed, queue_size=10)

        # 加载配置文件
        self.load_config()

    def set_angles(self, hand_joint):
        """
        设置手指关节角度
        Args:
            hand_joint: [little_finger, ring_finger, middle_finger, index_finger, thumb, thumb_rotation]
        """
        def grip_callback(msg):
            if msg:
                print("设置手指关节角度成功")
            else:
                print("设置手指关节角度失败")
            grip_control_ok_event.set()

        # 订阅是否设置成功
        rospy.Subscriber("/rm_driver/Set_Hand_Angle_Result", Bool, grip_callback)
        time.sleep(self.config["subscribe_seconds"])
        grip_control_ok_event = threading.Event()

        # 设置速度
        hand_speed = self.config['finger_speed']
        print("设置速度", hand_speed)
        hand_speed_msg = Hand_Speed()
        hand_speed_msg.hand_speed = hand_speed
        self.hand_speed_pub.publish(hand_speed_msg)
        time.sleep(self.config["finger_speed_publish_seconds"])

        # 设置角度
        print("设置手指关节角度为", hand_joint)
        hand_angle_msg = Hand_Angle()
        hand_angle_msg.hand_angle = hand_joint
        self.hand_angle_pub.publish(hand_angle_msg)
        time.sleep(self.config["finger_angle_publish_seconds"])

        # 等待设置完成
        grip_control_ok_event.wait()

    def load_config(self):
        with open('config.json') as f:
            self.config = json.load(f)


if __name__ == '__main__':
    hand = Hand()
    # hand.set_angles([0, 0, 0, 0, 0, 0])
    hand.set_angles([1000, 1000, 1000, 1000, 1000, 0])
    # rospy.spin()
    rospy.signal_shutdown("Program finished")  # 显式关闭 ROS 节点
    print("Program finished")