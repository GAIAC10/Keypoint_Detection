"""
原来: dataset -> images         -> 1.jpg/2.jpg/...
      dataset -> labelme_jsons  -> 1.json/2.json/...
现在: 5_labelme2yolo-keypoint.py

pip install seedir emoji
import seedir as sd
sd.seedir("table_format", style="emoji", depthlimit=2)
"""

import os
import shutil
import random
from tqdm import tqdm

Dataset_root = 'dataset'
os.chdir(os.path.join(Dataset_root, 'images'))
print(os.listdir())  # ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

"""
划分训练集和测试集
"""
test_frac = 0.2  # 测试集比例
random.seed(123)  # 随机数种子，便于复现
folder = '.'
img_paths = os.listdir(folder)
random.shuffle(img_paths)  # 随机打乱
val_number = int(len(img_paths) * test_frac)  # 测试集文件个数
train_files = img_paths[val_number:]         # 训练集文件名列表
val_files = img_paths[:val_number]           # 测试集文件名列表
print('数据集文件总数', len(img_paths))
print('训练集文件个数', len(train_files))
print('测试集文件个数', len(val_files))

"""
将训练集图像移动到images/train目录
"""
os.mkdir('train')
for each in tqdm(train_files):
    shutil.move(each, 'train')

"""
将测试集图像移动到images/val目录
"""
os.mkdir('val')
for each in tqdm(val_files):
    shutil.move(each, 'val')

"""
将训练集标注移动到lebalme_jsons/train目录
"""
os.chdir('../labelme_jsons')
os.mkdir('train')
for each in tqdm(train_files):
    srt_path = each.split('.')[0] + '.json'
    shutil.move(srt_path, 'train')

"""
将测试集标注移动到lebalme_jsons/val目录
"""
os.mkdir('val')
for each in tqdm(val_files):
    srt_path = each.split('.')[0] + '.json'
    shutil.move(srt_path, 'val')

os.chdir('../../')