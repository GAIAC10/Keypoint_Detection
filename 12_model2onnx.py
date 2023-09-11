"""
模型导出ONNX，Netron可视化

使用Netron可视化模型结构
Netron：https://netron.app
"""

from ultralytics import YOLO

# 载入预训练模型
model = YOLO('model/yolov8n-pose.pt')
# model = YOLO('yolov8s-pose.pt')
# model = YOLO('yolov8m-pose.pt')
# model = YOLO('yolov8l-pose.pt')
# model = YOLO('yolov8x-pose.pt')
# model = YOLO('yolov8x-pose-p6.pt')

# 将模型导出为onnx文件
success = model.export(format='onnx')
