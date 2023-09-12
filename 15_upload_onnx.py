"""
模型转ONNX、TensorRT、OpenVino、CoreML、TFLite、Paddle
模型部署：智能手机APP、开发板、浏览器、服务器、微信小程序

GPU版本
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
CPU版本
pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

验证安装配置成功
import onnxruntime
onnxruntime.get_device()
"""

# 使用onnx 效果 快好
# model -> onnx import 12_model2onnx.py
# 使用onnxruntime推理框架 效果 不好
# onnx -> onnxruntime
import cv2
import numpy as np
from PIL import Image
import onnxruntime
import torch
import matplotlib.pyplot as plt

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

kpts_shape = [3, 3]  # 关键点 shape

"""
创建ONNX Runtime的InferenceSession
"""
ort_session = onnxruntime.InferenceSession(
    'model/Triangle_215_yolov8m_pretrain.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

"""
输入一个随机Tensor，推理预测
"""
import torch

# 输入图片tensor的大小
x = torch.randn(1, 3, 640, 640).numpy()
# x.shape  (1, 3, 640, 640)
ort_inputs = {'images': x}
# 经过模型,输出图片的tensor大小
ort_output = ort_session.run(['output0'], ort_inputs)[0]
# ort_output.shape  (1, 14, 8400)

"""
获得ONNX模型输入层和数据维度
"""
model_input = ort_session.get_inputs()
input_name = [model_input[0].name]
# input_name  ['images']
input_shape = model_input[0].shape
# input_shape  [1, 3, 640, 640]
input_height, input_width = input_shape[2:]

"""
获得ONNX模型输出层和数据维度
"""
model_output = ort_session.get_outputs()
output_name = [model_output[0].name]
# output_name ['output0']
output_shape = model_output[0].shape
# output_shape  [1, 14, 8400]

"""
载入图像
"""
img_path = 'val_some/triangle_1.jpg'
# 导入 BGR 格式的图像
img_bgr = cv2.imread(img_path)

"""
预处理-缩放图像尺寸
"""
img_bgr_640 = cv2.resize(img_bgr, [input_height, input_width])
img_rgb_640 = img_bgr_640[:, :, ::-1]
# X 方向 图像缩放比例
x_ratio = img_bgr.shape[1] / input_width
# Y 方向 图像缩放比例
y_ratio = img_bgr.shape[0] / input_height

"""
预处理-构造输入张量
"""
# 预处理-归一化
input_tensor = img_rgb_640 / 255
# 预处理-构造输入 Tensor
input_tensor = np.expand_dims(input_tensor, axis=0)  # 加 batch 维度
input_tensor = input_tensor.transpose((0, 3, 1, 2))  # N, C, H, W
input_tensor = np.ascontiguousarray(input_tensor)  # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
input_tensor = torch.from_numpy(input_tensor).to(device).float()  # 转 Pytorch Tensor
# input_tensor = input_tensor.half() # 是否开启半精度，即 uint8 转 fp16，默认转 fp32
# input_tensor.shape  torch.Size([1, 3, 640, 640])

"""
执行推理预测
"""
# ONNX Runtime 推理预测
ort_output = ort_session.run(output_name, {input_name[0]: input_tensor.numpy()})[0]
# 转 Tensor
preds = torch.Tensor(ort_output)
# preds.shape  torch.Size([1, 14, 8400])

"""
后处理
"""
# 置信度阈值过滤、非极大值抑制NMS过滤
from ultralytics.utils import ops

preds = ops.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.7, nc=1)
pred = preds[0]
# pred.shape  torch.Size([3, 15])

"""
后处理数据 -> 目标检测数据
"""
pred_det = pred[:, :6].cpu().numpy()
# pred_det
# array([[1216, 435, 2546, 1762, 0.96564, 0],
#        [2894, 656, 4440, 1534, 0.9498, 0],
#        [1740, 1903, 2542, 3587, 0.92867, 0]], dtype=float32)
# 目标检测预测结果：左上角X、左上角Y、右下角X、右下角Y、置信度、类别ID

# num_bbox = len(pred_det)  预测出 {} 个框
num_bbox = len(pred_det)

# bboxes_cls = pred_det[:, 5]  类别

# bboxes_conf = pred_det[:, 4]  置信度

# 目标检测框 XYXY 坐标
# 还原为缩放之前原图上的坐标
pred_det[:, 0] = pred_det[:, 0] * x_ratio
pred_det[:, 1] = pred_det[:, 1] * y_ratio
pred_det[:, 2] = pred_det[:, 2] * x_ratio
pred_det[:, 3] = pred_det[:, 3] * y_ratio
bboxes_xyxy = pred_det[:, :4].astype('uint32')

"""
后处理数据 -> 关键点检测数据
"""
# 将缩放之后图像的预测结果，投射回原始尺寸
pred_kpts = pred[:, 6:].view(len(pred), kpts_shape[0], kpts_shape[1])
# pred_kpts.shape  torch.Size([3, 3, 3])
bboxes_keypoints = pred_kpts.cpu().numpy()
# 还原为缩放之前原图上的坐标
bboxes_keypoints[:, :, 0] = bboxes_keypoints[:, :, 0] * x_ratio
bboxes_keypoints[:, :, 1] = bboxes_keypoints[:, :, 1] * y_ratio
bboxes_keypoints = bboxes_keypoints.astype('uint32')

"""
OpenCV可视化关键点
"""
# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)  # 框的 BGR 颜色
bbox_thickness = 6  # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size': 4,  # 字体大小
    'font_thickness': 10,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': -80,  # Y 方向，文字偏移距离，向下为正
}

# 关键点 BGR 配色
kpt_color_map = {
    0: {'name': 'angle_30', 'color': [255, 0, 0], 'radius': 40},  # 30度角点
    1: {'name': 'angle_60', 'color': [0, 255, 0], 'radius': 40},  # 60度角点
    2: {'name': 'angle_90', 'color': [0, 0, 255], 'radius': 40},  # 90度角点
}

# 点类别文字
kpt_labelstr = {
    'font_size': 4,  # 字体大小
    'font_thickness': 10,  # 字体粗细
    'offset_x': 30,  # X 方向，文字偏移距离，向右为正
    'offset_y': 120,  # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [196, 75, 255], 'thickness': 12},  # 30度角点-60度角点
    {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [180, 187, 28], 'thickness': 12},  # 30度角点-90度角点
    {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [47, 255, 173], 'thickness': 12},  # 60度角点-90度角点
]

for idx in range(num_bbox):  # 遍历每个框
    # 获取该框坐标
    bbox_xyxy = bboxes_xyxy[idx]
    # 获取框的预测类别（对于关键点检测，只有一个类别）
    bbox_label = 'sjb_rect'
    # 画框
    img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                            bbox_thickness)
    # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
    img_bgr = cv2.putText(img_bgr, bbox_label,
                          (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                          cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color,
                          bbox_labelstr['font_thickness'])
    bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
    # 画该框的骨架连接
    for skeleton in skeleton_map:
        # 获取起始点坐标
        srt_kpt_id = skeleton['srt_kpt_id']
        srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        srt_kpt_y = bbox_keypoints[srt_kpt_id][1]
        # 获取终止点坐标
        dst_kpt_id = skeleton['dst_kpt_id']
        dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
        dst_kpt_y = bbox_keypoints[dst_kpt_id][1]
        # 获取骨架连接颜色
        skeleton_color = skeleton['color']
        # 获取骨架连接线宽
        skeleton_thickness = skeleton['thickness']
        # 画骨架连接
        img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                           thickness=skeleton_thickness)
    # 画该框的关键点
    for kpt_id in kpt_color_map:
        # 获取该关键点的颜色、半径、XY坐标
        kpt_color = kpt_color_map[kpt_id]['color']
        kpt_radius = kpt_color_map[kpt_id]['radius']
        kpt_x = bbox_keypoints[kpt_id][0]
        kpt_y = bbox_keypoints[kpt_id][1]
        # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
        img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
        # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        # kpt_label = str(kpt_id) # 写关键点类别 ID
        kpt_label = str(kpt_color_map[kpt_id]['name'])  # 写关键点类别名称
        img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                              kpt_labelstr['font_thickness'])
plt.imshow(img_bgr[:, :, ::-1])
plt.show()
