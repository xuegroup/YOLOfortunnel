# coding utf-8
# 作者：贾非
# 时间：2023/3/7 14:29
import datetime
import os
import time
import itertools
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
import torch
import yaml
import torch.distributed as dist


def test_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


def read_yaml(file):
    with open(file, 'r') as f:
        data = yaml.load(f, yaml.FullLoader)
    print(data)
    return data


def yield_cal(data):
    for i in data:
        start_time = time.time()
        yield i
        end_time = time.time()
        print('cal_time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    test_cuda()

    # yaml特点，定义了anchors后，在后面使用时，利用eval，yaml可自动将两者对应起来
    # da = read_yaml('cfg/yolov5l.yaml')
    # anchors = [1, 2, 3]
    # print(eval(da['head'][14][3][1]))
    # b = eval('False')
    # print(b)

    # img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)  # 灰度模式读取图像，添加一个维度，转为(1, h, w)
    # print(img.shape)
    # img = img[..., None].transpose(2, 0, 1)
    # print(img.shape)

    # a = [1, 2, 3]
    # print(type(a).__name__)
    # for i, data in enumerate(yield_cal(a)):
    #     a[i] += 100
    #     time.sleep(1)

    print(str(datetime.timedelta(seconds=3600)))

    # a = torch.tensor([[1, 2, 3, 4],
    #                   [6, 9, 1, 3]])
    # print(a.chunk(4, dim=-1))  # 维度保持不变

    # a = [[{'image_id': 0, 'cat_id': 1}, {'image_id': 2, 'cat_id': 3}], [{'image_id': 4, 'cat_id': 2}]]
    # print(list(itertools.chain.from_iterable(a)))  # 对每一个元素，只能去掉一个中括号

    # a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # i = torch.tensor([0, 1])
    # j = torch.tensor([1, 2])
    # print(a[i, j])
    # print(a[i, j, None])  # 添加维度
    # 下面两种输出维度不同，第一种会保持维度，类似keepdim
    # print(a[:, 2:3])
    # print(a[:, 2])

    # w = torch.tensor([0, 1, 2, 3])
    # c = torch.tensor([0, 1, 2])
    # x_ij, y_ij = torch.meshgrid(w, c, indexing='ij')
    # print(x_ij)
    # print(y_ij)
    # x_xy, y_xy = torch.meshgrid(w, c, indexing='xy')
    # print(x_xy)
    # print(y_xy)

    # a = '/my_yolo_dataset/train/images/2009_004012.jpg'
    # print(os.path.splitext(a))
    # print(a.replace('images', 'labels').replace(os.path.splitext(a)[-1], '.txt'))

