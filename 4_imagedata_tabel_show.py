"""
对图片数据进行规律总结并表格输出
"""

import os
import cv2
import numpy as np
import pandas as pd
import math
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

# 创建 table_format 文件夹，用于存放图表
if not os.path.exists('table_format'):
    os.mkdir('table_format')
    print('create dir table_format')

"""
在图表中显示中文

windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

Mac操作系统，参考 https://www.ngui.cc/51cto/show-727683.html
下载 simhei.ttf 字体文件
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf

# Linux操作系统，例如 云GPU平台：https://featurize.cn/?s=d7ce99f842414bfcaea5662a97581bd1
# 如果报错 Unable to establish SSL connection.，重新运行本代码块即可
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /environment/miniconda3/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf --no-check-certificate
!rm -rf /home/featurize/.cache/matplotlib
"""

import matplotlib
import matplotlib.pyplot as plt

"""
test 显示功能
"""
matplotlib.rc("font", family='SimHei')  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot([1, 2, 3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()

"""
可视化一些image
"""
folder_path = 'Triangle_215_Keypoint_Labelme/images'
N = 16  # 可视化图像的个数
# n 行 n 列
n = math.floor(np.sqrt(N))
# 读取文件夹中的所有图像
images = []
for each_img in os.listdir(folder_path)[:N]:
    img_path = os.path.join(folder_path, each_img)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)
# 画图
fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig,
                 111,  # 类似绘制子图 subplot(111)
                 nrows_ncols=(n, n),  # 创建 n 行 m 列的 axes 网格
                 axes_pad=0.02,  # 网格间距
                 share_all=True
                 )
# 遍历每张图像
for ax, im in zip(grid, images):
    ax.imshow(im)
    ax.axis('off')
plt.tight_layout()
plt.savefig('table_format/图像-一些图像.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
图片的尺寸分布
"""
# 载入csv文件
df = pd.read_csv('kpt_dataset_eda.csv')  # 1666*22
# 图像个数
print(len(df['imagePath'].unique()))  # 215张不同的图片
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
x = df['imageWidth']
y = df['imageHeight']
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.figure(figsize=(10,10))
plt.scatter(x, y, c=z,  s=5, cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
xy_max = max(max(df['imageWidth']), max(df['imageHeight']))
plt.xlim(xmin=0, xmax=xy_max)
plt.ylim(ymin=0, ymax=xy_max)
plt.ylabel('height', fontsize=25)
plt.xlabel('width', fontsize=25)
plt.savefig('table_format/图像-图像宽高尺寸分布.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
不同标注种类的个数
"""
df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df['label_type'].unique():
    label_type_list.append(each)
    num_list.append(len(df[df['label_type'] == each]))
df_num['label_type'] = label_type_list
df_num['num'] = num_list
df_num = df_num.sort_values(by='num', ascending=False)
print(df_num)  # rectangle/point/polygon个数
plt.figure(figsize=(22, 7))
x = df_num['label_type']
y = df_num['num']
plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')
plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('标注类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.savefig('table_format/图像-各标注种类个数.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
不用图片的标注个数
"""
df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df['imagePath'].unique():
    label_type_list.append(each)
    num_list.append(len(df[df['imagePath'] == each]))
df_num['label_type'] = label_type_list
df_num['num'] = num_list
df_num = df_num.sort_values(by='num', ascending=False)
plt.figure(figsize=(22, 10))
x = df_num['label_type']
y = df_num['num']
plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')
plt.xticks(rotation=90)
plt.tick_params(labelsize=8)
plt.xlabel('图像路径', fontsize=20)
plt.ylabel('标注个数', fontsize=20)
plt.savefig('table_format/图像-不同图像的标注个数.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
标注框-框中线点的位置分布
"""
df_box = df[df['label_type'] == 'rectangle']
df_box = df_box.reset_index(drop=True)
# 密集的地方有颜色
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
x = df_box['bbox_center_x_norm']
y = df_box['bbox_center_y_norm']
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.figure(figsize=(7,7))
plt.scatter(x, y,c=z,  s=3,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('x_center', fontsize=25)
plt.ylabel('y_center', fontsize=25)
plt.savefig('table_format/框标注-框中心点位置分布.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
标注框-框宽高分布
"""
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
x = df_box['bbox_width_norm']
y = df_box['bbox_height_norm']
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.figure(figsize=(7,7))
# plt.figure(figsize=(12,12))
plt.scatter(x, y,c=z,  s=1,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
plt.xlim(0,1.02)
plt.ylim(0,1.015)
plt.xlabel('width', fontsize=25)
plt.ylabel('height', fontsize=25)
plt.savefig('table_format/框标注-框宽高分布.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
关键点检测-不同类别的标注个数
"""
# 二选一运行
df_point = df[df['label_type']=='point']    # 所有关键点
# df_point = df[df['label']=='angle_30']      # 指定一类关键点
df_point = df_point.reset_index(drop=True)
df_num = pd.DataFrame()
label_type_list = []
num_list = []
for each in df_point['label'].unique():
    label_type_list.append(each)
    num_list.append(len(df_point['label'] == each))
df_num['label_type'] = label_type_list
df_num['num'] = num_list
df_num = df_num.sort_values(by='num', ascending=False)
print(df_num) # angle_30/angle_60/angle_90 个数
plt.figure(figsize=(22, 7))
x = df_num['label_type']
y = df_num['num']
plt.bar(x, y, facecolor='#1f77b4', edgecolor='k')
plt.xticks(rotation=90)
plt.tick_params(labelsize=15)
plt.xlabel('标注类别', fontsize=20)
plt.ylabel('图像数量', fontsize=20)
plt.savefig('table_format/关键点标注-不同类别点的标注个数.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
关键带检测-点的位置分布
"""
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
# 所有关键点
x = df_point['kpt_x_norm']
y = df_point['kpt_y_norm']
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.figure(figsize=(7,7))
plt.scatter(x, y,c=z,  s=3,cmap='Spectral_r')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
plt.tick_params(labelsize=15)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('kpt_x', fontsize=25)
plt.ylabel('kpt_y', fontsize=25)
plt.savefig('table_format/关键点标注-关键点位置分布.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
多段线标注-点的个数直方图
"""
df_poly = df[df['label_type'] == 'polygon']
df_poly = df_poly.reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.hist(df_poly['poly_num_points'], bins=20)
plt.savefig('table_format/多段线标注-多段线点个数直方图.pdf', dpi=120, bbox_inches='tight')
plt.show()

"""
多段线标注-区域面积直方图
"""
plt.figure(figsize=(10, 6))
plt.hist(df_poly['poly_area'], bins=40)
plt.savefig('table_format/多段线标注-多段线区域面积直方图.pdf', dpi=120, bbox_inches='tight')
plt.show()