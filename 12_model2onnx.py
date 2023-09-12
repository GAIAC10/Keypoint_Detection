"""
模型导出ONNX，Netron可视化
导出文档:https://docs.ultralytics.com/modes/export/#usage-examples

使用Netron可视化模型结构
Netron：https://netron.app
"""

"""
命令行
"""
# yolo export model=checkpoint/Triangle_215_yolov8l_pretrain.pt format=onnx

"""
api
"""
from ultralytics import YOLO

# 载入预训练模型
model = YOLO('model/Triangle_215_yolov8m_pretrain.pt')
# model = YOLO('yolov8s-pose.pt')
# model = YOLO('yolov8m-pose.pt')
# model = YOLO('yolov8l-pose.pt')
# model = YOLO('yolov8x-pose.pt')
# model = YOLO('yolov8x-pose-p6.pt')

# 将模型导出为onnx文件
success = model.export(format='onnx')

"""
验证onnx模型导出成功
"""
import onnx
# 读取 ONNX 模型
onnx_model = onnx.load('model/yolov8n-pose.onnx')
# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
print('无报错，onnx模型载入成功')

"""
使用onnxsim优化ONNX模型
"""
# Github主页：https://github.com/daquexian/onnx-simplifier
# 在线网页：https://convertmodel.com/#input=onnx&output=onnx
# 大缺弦老师讲解-ONNX 新特性和最佳实践介绍：https://www.bilibili.com/video/BV1LG411K7ah