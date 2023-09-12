"""
YOLO V8 Python API 图片
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device ---> {}'.format(device))

# 载入预训练模型
# model = YOLO('yolov8n-pose.pt')
# model = YOLO('yolov8s-pose.pt')
# model = YOLO('yolov8m-pose.pt')
# model = YOLO('yolov8l-pose.pt')
# model = YOLO('yolov8x-pose.pt')
model = YOLO('model/yolov8x-pose-p6.pt')

# 切换计算设备(将模型放在设备上)
model.to(device)
# 强制切换设备
# model.cpu()  # CPU
# model.cuda() # GPU
# 模型自带信息
print("model.device ---> {}".format(model.device))
# 模型的类别
print("model.names ---> {}".format(model.names))

"""
获得预测结果
"""
img_path = 'val_some/two_runners.jpg'
results = model(img_path, verbose=False)
print("results ---> {}".format(results))
# 一张图片对应一个结果
print("len(results) ---> {}".format(len(results)))  # 1
# 获取该图片的结果(list中)
print("results[0] ---> {}".format(results[0]))
"""
boxes(目标识别检测结果): Object
keypoints(关键点检测结果): Object
masks(实例分割检测结果): Object
names: {0: 'person'}
orig_img:array(xxx)
orig_shape:
path:
probs:
save_dir:
speed:
"""
"""
解析 目标检测 预测结果
"""
# 预测框的所有类别（MS COCO数据集八十类）
print("results[0].names ---> {}".format(results[0].names))

# 预测类别 ID
print("results[0].boxes.cls ---> {}".format(results[0].boxes.cls))
num_bbox = len(results[0].boxes.cls)
print('预测出 {} 个框'.format(num_bbox))

# 每个框的置信度
print("results[0].boxes.conf ---> {}".format(results[0].boxes.conf))

# 每个框的：左上角XY坐标、右下角XY坐标
print("results[0].boxes.xyxy ---> {}".format(results[0].boxes.xyxy))

# 转成整数的 numpy array(tensor -> array)
bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')
print("bboxes_xyxy ---> {}".format(results[0].boxes.xyxy))

"""
解析 关键点检测 预测结果
"""
# 每个框，每个关键点的 XY坐标 置信度
print("Keypoint.shape ---> {}".format(results[0].keypoints.shape))  # torch.Size([2, 17, 3]) 有2个框,每个框17个点,每个点(x, y, conf)
print("Keypoint ---> {}".format(results[0].keypoints))
# bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')
bboxes_keypoints = results[0].keypoints.data.cpu().numpy()
# 转为 numpy array
print("numpy_array_bboxes ---> {}".format(bboxes_keypoints))

"""
OpenCV可视化关键点
"""
img_bgr = cv2.imread(img_path)
plt.imshow(img_bgr[:, :, ::-1])
plt.show()

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)  # 框的 BGR 颜色
bbox_thickness = 6  # 框的线宽

# 框类别文字
bbox_labelstr = {
    'font_size': 6,  # 字体大小
    'font_thickness': 14,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': -80,  # Y 方向，文字偏移距离，向下为正
}

# 关键点 BGR 配色(YOLO V8关键点只有17个)
kpt_color_map = {
    0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': 25},  # 鼻尖
    1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': 25},  # 右边眼睛
    2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': 25},  # 左边眼睛
    3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': 25},  # 右边耳朵
    4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': 25},  # 左边耳朵
    5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': 25},  # 右边肩膀
    6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': 25},  # 左边肩膀
    7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': 25},  # 右侧胳膊肘
    8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': 25},  # 左侧胳膊肘
    9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': 25},  # 右侧手腕
    10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': 25},  # 左侧手腕
    11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': 25},  # 右侧胯
    12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': 25},  # 左侧胯
    13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': 25},  # 右侧膝盖
    14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': 25},  # 左侧膝盖
    15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': 25},  # 右侧脚踝
    16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': 25},  # 左侧脚踝
}

# 点类别文字
kpt_labelstr = {
    'font_size': 4,  # 字体大小
    'font_thickness': 10,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': 150,  # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 5},  # 右侧脚踝-右侧膝盖
    {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 5},  # 右侧膝盖-右侧胯
    {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 5},  # 左侧脚踝-左侧膝盖
    {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 5},  # 左侧膝盖-左侧胯
    {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 5},  # 右侧胯-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 5},  # 右边肩膀-右侧胯
    {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 5},  # 左边肩膀-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 5},  # 右边肩膀-左边肩膀
    {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 5},  # 右边肩膀-右侧胳膊肘
    {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 5},  # 左边肩膀-左侧胳膊肘
    {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 5},  # 右侧胳膊肘-右侧手腕
    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 5},  # 左侧胳膊肘-左侧手腕
    {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 5},  # 右边眼睛-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 5},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 5},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 5},  # 右边眼睛-右边耳朵
    {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 5},  # 左边眼睛-左边耳朵
    {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 5},  # 右边耳朵-右边肩膀
    {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 5}  # 左边耳朵-左边肩膀
]

for idx in range(num_bbox):  # 遍历每个框
    # 获取该框坐标
    bbox_xyxy = bboxes_xyxy[idx]
    # 获取框的预测类别（对于关键点检测，只有一个类别）
    bbox_label = results[0].names[0]
    """
    画框
    """
    img_bgr = cv2.rectangle(
        img_bgr,
        (
            bbox_xyxy[0],
            bbox_xyxy[1]
        ),
        (
            bbox_xyxy[2],
            bbox_xyxy[3]
        ),
        bbox_color,
        bbox_thickness
    )
    # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
    img_bgr = cv2.putText(
        img_bgr,
        bbox_label,
        (
            bbox_xyxy[0] + bbox_labelstr['offset_x'],
            bbox_xyxy[1] + bbox_labelstr['offset_y']
        ),
        cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'],
        bbox_color,
        bbox_labelstr['font_thickness']
    )
    bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度
    """
    画该框的骨架连接
    """
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
        img_bgr = cv2.line(
            img_bgr,
            (
                int(srt_kpt_x),
                int(srt_kpt_y)
            ),
            (
                int(dst_kpt_x),
                int(dst_kpt_y)
            ),
            color=skeleton_color,
            thickness=skeleton_thickness
        )
    """
    画该框的关键点
    """
    for kpt_id in kpt_color_map:
        # 获取该关键点的颜色、半径、XY坐标
        kpt_color = kpt_color_map[kpt_id]['color']
        kpt_radius = kpt_color_map[kpt_id]['radius']
        kpt_x = bbox_keypoints[kpt_id][0]
        kpt_y = bbox_keypoints[kpt_id][1]
        # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
        img_bgr = cv2.circle(
            img_bgr,
            (
                int(kpt_x),
                int(kpt_y)
            ),
            kpt_radius,
            kpt_color,
            -1
        )
        # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        kpt_label = str(kpt_id)  # 写关键点类别 ID（二选一）
        # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
        img_bgr = cv2.putText(
            img_bgr,
            kpt_label,
            (
                int(kpt_x) + kpt_labelstr['offset_x'],
                int(kpt_y) + kpt_labelstr['offset_y']
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            kpt_labelstr['font_size'], kpt_color,
            kpt_labelstr['font_thickness']
        )
plt.imshow(img_bgr[:, :, ::-1])
plt.show()
