#!/usr/bin/env python3
import json
import cv2
import rospy
from voice import Voice
from camera import Camera
from armControl import ArmControl
from hand import Hand


if __name__ == '__main__':
    rospy.init_node('d5', anonymous=True)

    with open('config.json') as f:
        config = json.load(f)
        
    voice = Voice()
    camera = Camera()
    arm = ArmControl()
    hand = Hand()


    # in_matrix, ex_matrix = camera.get_intrinsic_extrinsic_matrix()
    in_matrix = [
        [606.9666748,    0.,         327.58468628],
        [  0.,         605.09527588, 233.15785217],
        [  0.,           0.,           1.        ]
    ]

    ex_matrix = [
        [-0.46624163,  0.88301581, -0.0538685,  -0.2130471 ],
        [ 0.70870622,  0.33637183, -0.62015279,  0.39217146],
        [-0.52948487, -0.32731799, -0.78262935,  0.57819824],
        [ 0.,          0.,          0.,          1.        ]
 ]



    # voice.voice_translate()


    with open('temp.txt', 'r') as f:
        sentence = f.readline()
    bounding_boxes, depths = camera.get_bounding_box_from_sentence(sentence)

    # 如果字典中不存在某个键"key1"
    if "tomato" not in bounding_boxes or "tomato" not in depths:
        exit()


    x1, y1, x2, y2 = bounding_boxes["tomato"]
    img = cv2.imread(config['img_path'])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    cv2.imshow('Image with border', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    depth = depths["tomato"][int(v), int(u)] * 0.001  # 单位米
    x, y, z = camera.boundingbox_to_world_coordinate(u, v, depth, in_matrix, ex_matrix)
    arm.move_xyz(x, y, z)
    hand.set_angles([200, 300, 400, 500, 700, 0])
    hand.set_angles([100, 200, 300, 500, 500, 0])
