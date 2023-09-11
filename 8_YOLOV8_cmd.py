"""
安装YOLO V8
pip install ultralytics --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

检验安装成功
import ultralytics
ultralytics.checks()

安装第三方工具包
pip install numpy opencv-python pillow pandas matplotlib seaborn tqdm wandb seedir emoji -i https://pypi.tuna.tsinghua.edu.cn/simple

YOLO V8 命令行运行模板
yolo    task=detect         mode=train      model=yolov8[n/s/m/l/x].pt              source=0 [show是否实时展示在屏幕上]    decive=0/cpu
        classify            predict         yolov8[n/s/m/l/x]-cls.pt                source=0    decive=0/cpu
        segment(实力分割)   val             yolov8[n/s/m/l/x]-seg.pt                source=0    decive=0/cpu
        pose                export          yolov8[n/s/m/l/x/x-pose-p6]-pose.pt     source=0    decive=0/cpu
                track(多目标跟踪[task和mode共用])
                            benchmark

素材下载
图片:
https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/images/two_runners.jpg
https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/images/multi-person.jpeg
视频:
https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/videos/cxk.mp4
https://zihao-openmmlab.obs.myhuaweicloud.com/20220610-mmpose/videos/mother_wx.mp4
"""