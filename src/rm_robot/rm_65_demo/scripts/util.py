#!/usr/bin/env python

import re
import os
import cv2
import time
import math
import rospy
import atexit 
import paramiko
import numpy as np
import sounddevice as sd
import pyrealsense2 as rs
import scipy.io.wavfile as wav
from std_msgs.msg import Bool
from rm_msgs.msg import MoveJ_P, Plan_State, Hand_Angle, Hand_Speed

# 服务器信息
sftp_ip = '172.25.74.21'  # 替换为你的服务器IP地址
sftp_port = 22  # SFTP通常使用SSH端口22
username = 'zhangjs'  # SFTP用户名
password = 'zjs_server634'  # SFTP密码
remote_dir = '/home/zhangjs/GLIP/input_try1' # 远程目录路径
remote_txt_path = r'/home/zhangjs/GLIP/output_image/ii7_out_bbox.txt' # 服务器上的文件路径


def hand_grip(hand_angle, hand_speed, ori_x, ori_y, ori_z, ori_w):
    wait_for_ros_ok()

    hand_speed_pub = rospy.Publisher("/rm_driver/Hand_SpeedAngle", Hand_Speed, queue_size=10)
    hand_angle_pub = rospy.Publisher("/rm_driver/Hand_SetAngle",Hand_Angle, queue_size=10)
    time.sleep(2)
    hand_speed_msg = Hand_Speed()
    hand_angle_msg = Hand_Angle()
    hand_speed_msg.hand_speed = hand_speed
    hand_speed_pub.publish(hand_speed_msg)
    time.sleep(1)

    hand_angle_msg.hand_angle = hand_angle
    hand_angle_pub.publish(hand_angle_msg)

    time.sleep(2)
    publish_to_ros(
        -0.4,
        0,
        0.3,
        ori_x,
        ori_y,
        ori_z,
        ori_w,
        speed=0.07,
    )

    time.sleep(10)


    hand_angle_msg.hand_angle = [1000, 1000, 1000, 1000, 1000, 0]    
    hand_angle_pub.publish(hand_angle_msg)



    rospy.loginfo("finish hand grip")


    
    finish_ros_task()


def get_boundingbox(duration):
    def get_color_depth_image():
        # 初始化Realsense管道
        pipeline = rs.pipeline()
        config = rs.config()

        # 配置相机流（颜色流和深度流）
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动管道
        pipeline.start(config)

        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        pipeline.stop()
        cv2.destroyAllWindows()
        
        return color_image, depth_image

    def upload_img_to_server():
        # 本地文件列表
        local_files = [
            r"./temp.jpg",
            r"./temp.txt"
        ]

        local_txt_path = r"./ii7_out_bbox.txt"


        # 创建SSH客户端并连接到SFTP服务器
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(sftp_ip, sftp_port, username, password)

        # 打开SFTP客户端
        sftp = client.open_sftp()

        # 确保远程目录存在
        try:
            sftp.stat(remote_dir)
        except IOError:
            sftp.mkdir(remote_dir)

        # 上传多个文件
        for local_file in local_files:
            # 获取文件名
            file_name = os.path.basename(local_file)  # 使用 os.path.basename()
            # 构建远程文件路径
            remote_file_path = f'{remote_dir}/{file_name}'

            # 上传文件
            try:
                sftp.put(local_file, remote_file_path)
                print(f"文件 {file_name} 上传成功")
            except Exception as e:
                print(f"上传文件 {file_name} 失败: {e}")

        # 关闭SFTP和SSH连接
        sftp.close()
        client.close()
        print("文件上传成功")

        print(f"等待 {duration} 秒...")
        time.sleep(duration)




        # 创建SSH客户端并连接到SFTP服务器
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(sftp_ip, sftp_port, username, password)

        # 打开SFTP客户端
        sftp = client.open_sftp()

        # 下载文件
        sftp.get(remote_txt_path, local_txt_path)

        # 关闭SFTP和SSH连接
        sftp.close()
        client.close()
        print("文件下载成功")
        
        return

    def extract_bounding_boxes(file_path, target_label):
        top_left_coords = []  # 存储左上角坐标
        bottom_right_coords = []  # 存储右下角坐标

        # 打开并读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 遍历每一行，提取目标Label的BoundingBox
        for line in lines:
            match = re.match(r"Label: (\d+), Score: [^,]+, BoundingBox: \(([^)]+)\), \(([^)]+)\)", line)
            if match:
                label = int(match.group(1))
                if label == target_label:
                    # 解析左上角坐标
                    top_left = tuple(map(float, match.group(2).strip("()").split(',')))
                    top_left_coords.append(top_left)
                    # 解析右下角坐标
                    bottom_right = tuple(map(float, match.group(3).strip("()").split(',')))
                    bottom_right_coords.append(bottom_right)

        return top_left_coords, bottom_right_coords
    
    
    color_image, depth_image = get_color_depth_image()

    # 存储彩色图片
    jpg_path = r"./temp.jpg"
    if os.path.exists(jpg_path):
        os.remove(jpg_path)
    cv2.imwrite(jpg_path, color_image)
    print("等待图片储存")
    time.sleep(2)

    upload_img_to_server()

    file_path = r"./ii7_out_bbox.txt"  # 替换为你的文件路径

    target_label = 1  # 替换为你想要的Label
    top_left_boxes, bottom_right_boxes = extract_bounding_boxes(file_path, target_label)

    # 目标的bounding box坐标为 (x1, y1) 到 (x2, y2)
    x1, y1, x2, y2 = top_left_boxes[0][0], top_left_boxes[0][1], bottom_right_boxes[0][0], bottom_right_boxes[0][1]

    u = (x1 + x2) / 2
    v = (y1 + y2) / 2

    # 计算bounding box中心的深度
    depth = depth_image[int(v), int(u)] * 0.001  # 单位米
    
    return u, v, depth


def voice_translate(seconds, model="medium"):
    def record_audio(duration, output_file):
        """
        录制音频并保存为 MP3 文件
        :param duration: 录音时长（秒）
        :param output_file: 保存的文件名
        """
        try:
            print(f"开始录音... 持续 {duration} 秒")
            fs = 44100  # 采样率
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
            sd.wait()  # 等待录音完成
            wav.write(output_file, fs, audio_data)
            print(f"录音完成，保存为 {output_file}")

        except Exception as e:
            print(f"录音失败: {e}")

    def run_whisper(input_file, language, output_dir, output_format, model, task):
        """
        执行 whisper 命令
        :param model_dir: 模型路径
        :param input_file: 音频文件路径
        :param language: 语言代码（如 'zh'）
        :param output_dir: 输出目录
        :param output_format: 输出格式（如 'txt'）
        :param model: 模型类型（如 'medium'）
        :param task: 任务类型（如 'translate'）
        """
        try:
            command = (
                f"whisper {input_file} --language {language} "
                f"--output_dir {output_dir} --output_format {output_format} "
                f"--model {model} --task {task}"
            )
            print(f"运行命令: {command}")
            os.system(command)
            print("Whisper 执行完成")
        except Exception as e:
            print(f"运行 whisper 时出错: {e}")


    # 设置录音时长和文件名
    output_file = r"./temp.wav"

    # 录音并保存为 MP3
    if os.path.exists(output_file):
        os.remove(output_file)
    record_audio(seconds, output_file)

    # 执行 whisper 命令
    run_whisper(
        input_file=output_file,
        language="zh",
        output_dir=r".",
        output_format="txt",
        model=model,
        task="translate"
    )


def publish_to_ros(pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, ori_w, speed):
    def plan_state_callback(msg):
        if msg.state:
            rospy.loginfo("*******Plan State OK")
            finish_ros_task()
        else:
            rospy.loginfo("*******Plan State Fail")
        

    # 发布
    pub = rospy.Publisher('/rm_driver/MoveJ_P_Cmd', MoveJ_P, queue_size=10)

    # 订阅
    rospy.Subscriber("/rm_driver/Plan_State", Plan_State, plan_state_callback)

    rospy.sleep(1.0)

    moveJ_P_TargetPose = MoveJ_P()
    moveJ_P_TargetPose.Pose.position.x = pos_x
    moveJ_P_TargetPose.Pose.position.y = pos_y
    moveJ_P_TargetPose.Pose.position.z = pos_z
    moveJ_P_TargetPose.Pose.orientation.x = ori_x
    moveJ_P_TargetPose.Pose.orientation.y = ori_y
    moveJ_P_TargetPose.Pose.orientation.z = ori_z
    moveJ_P_TargetPose.Pose.orientation.w = ori_w
    moveJ_P_TargetPose.speed = speed

    # 发送指令
    rospy.loginfo("Publishing d5_move_arm by python...")
    pub.publish(moveJ_P_TargetPose)


def get_wxyz_from_rpy(roll, pitch, yaw):
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


def get_end_from_obj(obj_x, obj_y, obj_z, beta, hand_length, height, offset, hand_z_rotation):
    def sin(deg):
        ret_deg = math.sin(math.radians(deg))
        return ret_deg

    def cos(deg):
        ret_deg = math.cos(math.radians(deg))
        return ret_deg
    
    def rpy_after_z_rotation(r, p, y, a):
        # 转换为弧度
        r, p, y, a = np.radians([r, p, y, a])
        
        # 构造旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])
        R_y = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        R_z = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])
        R_add = np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])
        
        # 计算新矩阵
        R_initial = R_z @ R_y @ R_x
        R_new = R_initial @ R_add
        
        # 提取新的RPY
        new_pitch = -np.arcsin(R_new[2, 0])
        new_roll = np.arctan2(R_new[2, 1], R_new[2, 2])
        new_yaw = np.arctan2(R_new[1, 0], R_new[0, 0])
        
        return np.degrees(new_roll), np.degrees(new_pitch), np.degrees(new_yaw)

    # 异常处理
    if not(beta >= 0 and beta <= 90) or not(height > 0) or not (hand_length > 0):
        return None


    # 物体与世界坐标系x轴正方向的夹角，[-180°, 180°]
    theta = math.degrees(math.atan2(obj_y, obj_x))

    # 末端位置
    end_world_x = (obj_x - hand_length * cos(beta) * cos(theta)) + offset * cos(theta - 90)
    end_world_y = (obj_y - hand_length * cos(beta) * sin(theta)) + offset * sin(theta - 90)
    end_world_z = obj_z + height + hand_length * sin(beta)

    # 末端姿态（RPY角）
    end_roll = 0
    end_pitch = -(90 + beta)
    end_yaw = 180 + theta

    # 沿新z轴转hand_z_rotation
    end_roll, end_pitch, end_yaw = rpy_after_z_rotation(end_roll, end_pitch, end_yaw, hand_z_rotation)

    return end_world_x, end_world_y, end_world_z, end_roll, end_pitch, end_yaw


def get_world_coords_from_boundingbox(u, v, depth, camera_matrix, extrinsic_matrix):
    # 像素坐标 -> 相机坐标
    pixel_coords = np.array([u, v, 1])
    cam_coords = depth * np.linalg.inv(camera_matrix) @ pixel_coords

    # 相机坐标 -> 世界坐标
    cam_coords_h = np.append(cam_coords, 1)  # 转为齐次坐标
    world_coords_h = np.linalg.inv(extrinsic_matrix) @ cam_coords_h

    # 转为非齐次坐标
    world_coords = world_coords_h[:3] / world_coords_h[3]

    return world_coords


def get_camera_extrinsic_matrix():
    # Distortion coefficients (set to zero for simplicity)
    DIST_COEFFS = np.array([[0, 0, 0, 0]])

    # 3D world coordinate system points
    POINT_3D_LIST = np.float32([(-0.5, -0.15, 0), (-0.5, 0.15, 0), 
                                (-0.3, 0.15, 0), (-0.3, -0.15, 0)])

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    time.sleep(2)

    # Mouse click event handler
    POINT_2D_LIST = []
    def mouse_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            POINT_2D_LIST.append([x, y])

    # Clean-up function to stop pipeline
    @atexit.register
    def clean():
        pipeline.stop()
        device = profile.get_device()
        device.hardware_reset()
        time.sleep(2)

    # Get camera intrinsics
    profile_t = profile.get_stream(rs.stream.color)
    intr = profile_t.as_video_stream_profile().get_intrinsics()
    CAMERA_MATRIX = np.array([[intr.fx, 0, intr.ppx], 
                            [0, intr.fy, intr.ppy], 
                            [0, 0, 1]])

    while True:
        frame = pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
                                                                # 0 - - - 1
        cv2.imshow("video", color_image)                         # |       |
        cv2.setMouseCallback("video", mouse_event, color_image)  # 3 - - - 2

        if len(POINT_2D_LIST) == 4:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    # SolvePnP to get rvec and tvec
    retval, rvec, tvec = cv2.solvePnP(POINT_3D_LIST, np.float32(POINT_2D_LIST), CAMERA_MATRIX, DIST_COEFFS)

    # Convert rvec to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Build extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rmat
    extrinsic_matrix[:3, 3] = tvec.flatten()

    print(f"内参矩阵:\n{CAMERA_MATRIX}")
    print(f"外参矩阵:\n{extrinsic_matrix}")

    return CAMERA_MATRIX, extrinsic_matrix


def wait_for_ros_ok():
    global is_Plan_State_ok
    while not is_Plan_State_ok:
        time.sleep(0.5)
    is_Plan_State_ok = False


def finish_ros_task():
    global is_Plan_State_ok
    is_Plan_State_ok = True

###########################################################
# 全局变量
is_Plan_State_ok = False # 每个ros函数是否实行完它的功能