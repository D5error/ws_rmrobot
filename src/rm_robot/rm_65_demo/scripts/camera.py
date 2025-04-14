import re
import os
import cv2
import time
import atexit 
import numpy as np
import pyrealsense2 as rs
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 设置字体为常见的字体之一，例如 Arial
plt.rcParams['font.family'] =  'Ubuntu'

# 或者直接指定字体大小
plt.rcParams['font.size'] = 12  # 调整为适合的大小

class Camera:
    def __init__(self):
        # default: Load the model on the available device(s)
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
        # )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def analyze_sentence(self, sentence):
        """
        分析句子，提取关键物体名称。
        """
        prompt = f"从句子中提取关键物体名称：'{sentence}'。返回格式为：['物体1', '物体2']。"
        messages = [
            {
                "role": "robot",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        # inputs = self.processor(
        #     text=[prompt],
        #     return_tensors="pt"
        # )
        # inputs = inputs.to("cuda")
        # generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        # output_text = self.processor.batch_decode(
        #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        print(f"模型输出：{output_text[0]}")
        objects = eval(output_text[0])  # 假设模型输出是符合Python列表格式的字符串
        return objects
    

    # 获取RGBD图像
    def get_color_depth_image(self):
        # 初始化RealSense管道
        pipeline = rs.pipeline()
        config = rs.config()

        # 配置相机流（颜色流和深度流）
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # # 启动管道
        # profile = pipeline.start(config)
        # device = profile.get_device()
        # device.hardware_reset()

        try:
            # 启动管道
            profile = pipeline.start(config)
            print("Pipeline started successfully")
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            return None, None
        
        device = profile.get_device()
        print(f"Connected device: {device.get_info(rs.camera_info.name)}")

        device.hardware_reset()

        # 获取帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # pipeline.stop()
        # cv2.destroyAllWindows()
        
        return color_image, depth_image


    # 获取bounding box的坐标
    def get_coords(self, object_name, img_path):
        # description = f'return "{object_name}" coords. return format: "[left_x,top_y,right_x,bottom_y]" (integer coordinates) or "null". Please follow the format strictly, do not json.'

        description = f"""
            Return the bounding box coordinates of the object "{object_name}" in the image.
            The coordinates should be in the format: [left_x,top_y,right_x,bottom_y] where:
            - left_x: the x-coordinate of the left edge of the bounding box
            - top_y: the y-coordinate of the top edge of the bounding box
            - right_x: the x-coordinate of the right edge of the bounding box
            - bottom_y: the y-coordinate of the bottom edge of the bounding box

            All coordinates should be integer values. If the object is not found, return "null".
            Please follow the format strictly and do not use JSON.
        """
        messages = [
            {
                "role": "robot",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {
                        "type": "text", 
                        "text": description
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 获取模型输出的boundbox坐标，并可视化到图像上
        coords = output_text[0].strip('[]')
        if coords == "null":
            print(f"没有检测到{object_name}")
            return None
        
        # 绘制图像和边框
        # print(f"检测到{object_name}，bounding box坐标为：{coords}")
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(img)
        x1, y1, x2, y2 = map(int, coords.split(","))
        # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        # plt.gca().add_patch(rect)
        # plt.show()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Image with border', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        return x1, y1, x2, y2


    # 从句子中提取物体并获取其标定框坐标
    def get_bounding_box_from_sentence(self, sentence):
        # 分析句子，提取关键物体名称
        objects = self.analyze_sentence(sentence)
        print(f"提取到的物体名称：{objects}")

        # 获取RGB图像
        color_image, depth_image = self.get_color_depth_image()

        # 存储彩色图片
        jpg_path = r"./temp.jpg"
        if os.path.exists(jpg_path):
            os.remove(jpg_path)
        cv2.imwrite(jpg_path, color_image)

        # 获取每个物体的标定框坐标
        bounding_boxes = {}
        for obj in objects:
            coords = self.get_coords(obj, jpg_path)
            if coords:
                bounding_boxes[obj] = coords
        return bounding_boxes


    # 坐标转换为世界坐标
    def boundingbox_to_world_coordinate(self, u, v, depth, camera_matrix, extrinsic_matrix):
        # 像素坐标 -> 相机坐标
        pixel_coords = np.array([u, v, 1])
        cam_coords = depth * np.linalg.inv(camera_matrix) @ pixel_coords

        # 相机坐标 -> 世界坐标
        cam_coords_h = np.append(cam_coords, 1)  # 转为齐次坐标
        world_coords_h = np.linalg.inv(extrinsic_matrix) @ cam_coords_h

        # 转为非齐次坐标
        world_coords = world_coords_h[:3] / world_coords_h[3]

        return world_coords

    # 获取内参和外参矩阵
    def get_intrinsic_extrinsic_matrix(self):
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


if __name__ == "__main__":
    camera = Camera()
    # camera.get_intrinsic_extrinsic_matrix()
    sentence = ""
    bounding_boxes = camera.get_bounding_box_from_sentence(sentence)
    print("Bounding boxes:", bounding_boxes)

    # # 目标的bounding box坐标为 (x1, y1) 到 (x2, y2)
    # x1, y1, x2, y2 = get_coords()
    # u = (x1 + x2) / 2
    # v = (y1 + y2) / 2

    # # 计算bounding box中心的深度
    # depth = depth_image[int(v), int(u)] * 0.001  # 单位米
    
    # return u, v, depth