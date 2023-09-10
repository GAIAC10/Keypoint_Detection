"""
使用pandas对单张图片Key_Value进行展示
对Imagedata特征进行提取
ex:
在点多的地方添加anchor
对该地方进行特制算法
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# 创建一个空的pandas表格
df = pd.DataFrame()

labelme_path = "dataset/1.json"
with open(labelme_path, 'r', encoding='utf-8') as f:
    labelme = json.load(f)

imagePath = labelme['imagePath']
imageWidth = labelme['imageWidth']
imageHeight = labelme['imageHeight']
for each_ann in tqdm(labelme['shapes']):  # 遍历每一个标注
    df_temp = {}
    # 图像信息
    os_path = "dataset"
    df_temp['imagePath'] = imagePath
    df_temp['imageWidth'] = imageWidth
    df_temp['imageHeight'] = imageHeight
    """
    筛选出框标注
    """
    if each_ann['shape_type'] == 'rectangle':
        # 框的两点坐标
        bbox_keypoints = each_ann['points']
        bbox_keypoint_A_xy = bbox_keypoints[0]
        bbox_keypoint_B_xy = bbox_keypoints[1]
        # 左上角坐标
        bbox_top_left_x = int(min(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_top_left_y = int(min(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
        # 右下角坐标
        bbox_bottom_right_x = int(max(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_bottom_right_y = int(max(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
        # 标注信息
        df_temp['label_type'] = each_ann['shape_type']
        df_temp['label'] = each_ann['label']
        # 框坐标信息
        df_temp['bbox_top_left_x'] = bbox_top_left_x
        df_temp['bbox_top_left_y'] = bbox_top_left_y
        df_temp['bbox_bottom_right_x'] = bbox_bottom_right_x
        df_temp['bbox_bottom_right_y'] = bbox_bottom_right_y
        df_temp['bbox_width_pix'] = bbox_bottom_right_x - bbox_top_left_x
        df_temp['bbox_height_pix'] = bbox_bottom_right_y - bbox_top_left_y
        df_temp['bbox_width_norm'] = df_temp['bbox_width_pix'] / df_temp['imageWidth']
        df_temp['bbox_height_norm'] = df_temp['bbox_height_pix'] / df_temp['imageHeight']
        df_temp['bbox_center_x_pix'] = (bbox_top_left_x + bbox_bottom_right_x) / 2
        df_temp['bbox_center_y_pix'] = (bbox_top_left_y + bbox_bottom_right_y) / 2
        df_temp['bbox_center_x_norm'] = (bbox_top_left_x + bbox_bottom_right_x) / 2 / df_temp['imageWidth']
        df_temp['bbox_center_y_norm'] = (bbox_top_left_y + bbox_bottom_right_y) / 2 / df_temp['imageHeight']
    """
    筛选出关键点标注
    """
    if each_ann['shape_type'] == 'point':
        # 该点的 XY 坐标
        kpt_xy = each_ann['points'][0]
        kpt_x, kpt_y = int(kpt_xy[0]), int(kpt_xy[1])
        # 标注信息
        df_temp['label_type'] = each_ann['shape_type']
        df_temp['label'] = each_ann['label']
        # 关键点坐标信息
        df_temp['kpt_x_pix'] = kpt_x
        df_temp['kpt_y_pix'] = kpt_y
        df_temp['kpt_x_norm'] = kpt_x / df_temp['imageWidth']
        df_temp['kpt_y_norm'] = kpt_y / df_temp['imageHeight']
    """
    筛选出多段线（polygon）标注
    """
    if each_ann['shape_type'] == 'polygon':
        poly_points = np.array(each_ann['points']).astype('uint32').tolist()  # 该多段线每个点的坐标
        poly_num_points = len(poly_points)  # 该多段线点的个数
        # 计算多段线区域面积
        poly_pts = [np.array(each_ann['points'], np.int32).reshape((-1, 1, 2))]  # 该多段线每个点的坐标
        img_bgr = cv2.imread(os.path.join(os_path, imagePath))
        img_zeros = np.zeros(img_bgr.shape[:2], dtype='uint8')
        img_mask = cv2.fillPoly(img_zeros, poly_pts, 1)
        poly_area = np.sum(img_mask)
        # 标注信息
        df_temp['label_type'] = each_ann['shape_type']
        df_temp['label'] = each_ann['label']
        # 多段线信息
        df_temp['poly_points'] = poly_points
        df_temp['poly_num_points'] = poly_num_points
        df_temp['poly_area'] = poly_area
    # pandas 1.0
    # df = df.append(df_temp, ignore_index=True)
    # pandas 2.0
    df = pd.concat([df, pd.DataFrame([df_temp])], ignore_index=True)
# 所有的key
# print(df.columns)
# 表格形式
# print(df)