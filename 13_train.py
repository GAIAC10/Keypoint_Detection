"""
train自己的数据集

1.配置YOLO V8环境

2.安装wandb(模型训练监控)
1.在命令行中运行wandb login
2.按提示复制粘贴API Key至命令行中
fbe09e4f82c302d6687fc48867a6aa0e6eb77ad3

3.数据集下载
https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/Triangle_215_Keypoint_YOLO.zip

4.下载yaml文件
https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/Triangle_215.yaml

5.开始训练
https://docs.ultralytics.com/modes/train

YOLOV8-关键点检测预训练模型
yolov8n-pose.pt/yolov8s-pose.pt/yolov8m-pose.pt/yolov8l-pose.pt/yolov8x-pose.pt/yolov8x-pose-p6.pt

重要的训练参数
model YOLOV8模型(在原有模型的基础上进行训练)
data 配置文件（.yaml格式）
pretrained 是否在预训练模型权重基础上迁移学习泛化微调
epochs 训练轮次，默认100
batch batch-size，默认16
imgsz 输入图像宽高尺寸，默认640
device 计算设备（device=0 或 device=0,1,2,3 或 device=cpu）
project 项目名称，建议同一个数据集取同一个项目名称
name 实验名称，建议每一次训练对应一个实验名称
optimizer 梯度下降优化器，默认'SGD'，备选：['SGD', 'Adam', 'AdamW', 'RMSProp']
close_mosaic 是否关闭马赛克图像扩增，默认为0，也就是开启马赛克图像扩增
cls 目标检测分类损失函数cls_loss权重，默认0.5
box 目标检测框定位损失函数box_loss权重，默认7.5
dfl 类别不均衡时Dual Focal Loss损失函数dfl_loss权重，默认1.5。
DFL损失函数论文：Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
DFL损失函数知乎博客：https://zhuanlan.zhihu.com/p/147691786
pose 关键点定位损失函数pose_loss权重，默认12.0（只在关键点检测训练时用到）
kobj 关键点置信度损失函数keypoint_loss权重，默认2.0（只在关键点检测训练时用到）
"""

