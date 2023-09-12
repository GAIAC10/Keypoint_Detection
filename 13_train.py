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
参数详细网站
https://docs.ultralytics.com/modes/train

YOLOV8-关键点检测预训练模型
yolov8n-pose.pt     640
yolov8s-pose.pt     640
yolov8m-pose.pt     640
yolov8l-pose.pt     640
yolov8x-pose.pt     640
yolov8x-pose-p6.pt  1280

重要的训练参数
model           YOLO V8模型
data            配置文件（.yaml格式）
pretrained      是否在预训练模型权重基础上迁移学习泛化微调(保留骨架,True继续(Fine Tuning)|False重新训练(From Scratch))
epochs          训练轮次，默认100
batch           batch-size，默认16
imgsz           输入图像宽高尺寸，默认640
device          计算设备（device=0 或 device=0,1,2,3 或 device=cpu）
project         目标项目名称，建议同一个数据集取同一个项目名称( = run)
name            目标实验名称，建议每一次训练对应一个实验名称( = run/test1|test2|...)
optimizer       梯度下降优化器，默认'SGD'，备选：['SGD', 'Adam', 'AdamW', 'RMSProp']
close_mosaic    是否关闭马赛克图像扩增，默认为0，也就是开启马赛克图像扩增
# 目标检测
cls             目标检测分类损失函数cls_loss权重，默认0.5
box             目标检测框定位损失函数box_loss权重，默认7.5
dfl             类别不均衡时Dual Focal Loss损失函数dfl_loss权重，默认1.5。
                DFL损失函数论文：Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
                DFL损失函数知乎博客：https://zhuanlan.zhihu.com/p/147691786
# 关键点检测
pose            关键点定位损失函数pose_loss权重，默认12.0（只在关键点检测训练时用到）
kobj            关键点置信度损失函数keypoint_loss权重，默认2.0（只在关键点检测训练时用到）

重要的预测参数
- **基础参数**
model           YOLO V8模型
source          输入模型预测的图像、视频、文件夹路径
device          计算设备（device=0 或 device=0,1,2,3 或 device=cpu）
verbose         是否在命令行输出每一帧的预测结果，默认为True
- **后处理参数**
conf            目标检测置信度，默认0.25
iou             对同类框，目标检测非极大值抑制（NMS）的IOU阈值，默认0.7，越小，NMS越强
agnostic_nms    对所有类的框，而不仅仅对同类框，做目标检测非极大值抑制（NMS）
classes         只显示哪个或哪些类别的预测结果（class=0 或 class=[0,2,3]）
- **可视化参数**
show             是否将预测结果显示在屏幕上，默认为False。摄像头实时预测时，建议开启
show_labels      是否显示目标检测框类别文字，默认为True
show_conf        是否显示目标检测框置信度，默认为True
line_thickness   目标检测框线宽，默认为3
max_det          一张图像最多预测多少个框，默认为300
boxes            是否在实例分割预测结果中显示目标检测框
visualize        可视化模型特征
retina_masks     是否显示高分辨率实例分割mask，默认为False
- **保存结果参数**
save             保存预测结果可视化图像，默认为False
save_txt         将预测结果保存为txt文件，默认为False
save_conf        保存的预测结果中，包含置信度，默认为False
save_crop        保存的预测结果可视化图像中，是否裁切，默认为False
- **其它参数**
half             是否开启半精度（FP16），默认为False(内存降低)
augment          对输入数据做数据扩增，默认False(效果增强)
"""

"""
训练cmd
"""
# yolov8n-pose模型，迁移学习微调
# yolo pose train data=Triangle_215.yaml model=yolov8n-pose.pt pretrained=True project=Triangle_215_Test \
# name=n_pretrain epochs=50 batch=16 device=0

# yolov8n-pose模型，随机初始权重，从头重新学习
# yolo pose train data=Triangle_215.yaml model=yolov8n-pose.pt project=Triangle_215_Test name=n_scratch \
# epochs=50 batch=16 device=0

# 下载训练好的模型(自己数据集训练)
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8n_pose_pretrain.pt
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8s_pretrain.pt
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8m_pretrain.pt
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8l_pretrain.pt
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8x_pretrain.pt
# https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Triangle_215_Dataset/checkpoint/Triangle_215_yolov8x_p6_pretrain.pt

# 验证数据集
# 图片
# https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/triangle_dataset/test_img/triangle_1.jpg
# https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/triangle_dataset/test_img/triangle_2.jpg
# 视频
# https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/triangle_dataset/videos/triangle_6.mp4
# https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/triangle_dataset/videos/triangle_7.mp4