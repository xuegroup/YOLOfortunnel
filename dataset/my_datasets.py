# coding utf-8
# 作者：贾非
# 时间：2023/3/9 21:36
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class LoadImagesAndLabels(Dataset):

    def __init__(self, path, img_size=640, rank=-1):

        with open(path, 'r') as f:
            f = f.read().splitlines()
        self.path = path
        self.img_files = [x for x in f]
        self.img_files.sort()  # 防止不同系统排序不同
        n = len(self.img_files)
        self.img_size = img_size  # 输入模型的图像尺寸，也就是调整后的尺寸
        # (/my_yolo_dataset/train/images/2009_004012.jpg) -> (/my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        # 读取图像shape文件，为coco计算mAP需要的真实值做准备
        sp = path.replace('.txt', '.shapes')
        try:
            with open(sp, 'r') as f:
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == n, 'shapes lost'
        except Exception as e:
            if rank in [-1, 0]:
                img_files = tqdm(self.img_files, desc='reading img shape')
            else:
                img_files = self.img_files
            s = [Image.open(im).size for im in img_files]  # (w, h)
            np.savetxt(sp, s, fmt='%g')  # 一种存储方式，必须是纯数组，也可以用os中write
        # 记录图像原始shape
        self.shapes = np.array(s, dtype=np.float64)

        # 缓存标签文件, [cls, x, y, w, h]中心点坐标与宽高，坐标均是相对坐标
        # 这里设为(0，5)首先是二维数据，其次每张图像目标均不同，设为0让其自动填充
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n

        labels_loaded = False
        # 因为是纯数组，存储为npy格式，方便读取
        np_labels_path = os.path.dirname(self.label_files[0]) + '.cache.npy'
        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)  # 因为是序列化数据，allow_pickle需要置为True
            if len(x) == n:
                if rank in [-1, 0]:
                    print('reading labels from {}'.format(np_labels_path))
                self.labels = x
                labels_loaded = True

        if not labels_loaded:
            # 只在主进程显示
            if rank in [-1, 0]:
                pbar = tqdm(self.label_files, desc='reading labels')
            else:
                pbar = self.label_files
            for i, file in enumerate(pbar):
                with open(file, 'r') as f:
                    label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                # 将标签文件添加其中，准备写入npy文件
                self.labels[i] = label
            print('saving labels to {} for future fast loading'.format(np_labels_path))
            np.save(np_labels_path, self.labels)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img, (h0, w0), (h, w), (dw, dh) = load_image(self, index)
        shapes = (h0, w0)  # for COCO mAP rescaling

        # (h, w) -> (h, w, 1) -> (1, h, w)保证模型输入格式
        img = img[..., None].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        x = self.labels[index]  # np类型
        labels = x.copy()
        labels[:, 1] = (x[:, 1] * w) / (w + dw)
        labels[:, 2] = (x[:, 2] * h) / (h + dh)
        labels[:, 3] = (x[:, 3] * w) / (w + dw)
        labels[:, 4] = (x[:, 4] * h) / (h + dh)

        nl = len(labels)
        labels_out = torch.zeros((nl, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        # 为cocotools统计标签信息准备，不进行任何处理
        ori_shapes = self.shapes[index][::-1]  # (w, h) -> (h, w)
        x = self.labels[index]
        labels = x.copy()
        return torch.from_numpy(labels), ori_shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add targets img_index for build_targets()
        #           (b, 1, h, w)                (nt, 6)          (b)    (b)    (b)
        return torch.stack(img, dim=0), torch.cat(label, dim=0), path, shapes, index


def load_image(self, index, color=(114, 114, 114)):
    path = self.img_files[index]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 灰度模式读取图像
    assert img is not None, '{} not found'.format(path)
    h0, w0 = img.shape[:2]
    r = self.img_size / max(h0, w0)  # 将较大边缩至训练所需尺寸
    new_w, new_h = int(w0 * r), int(h0 * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # 等比例缩放
    dw, dh = self.img_size - new_w, self.img_size - new_h
    img = cv2.copyMakeBorder(img, top=0, bottom=dh, left=0, right=dw,
                             borderType=cv2.BORDER_CONSTANT, value=color)  # 加灰条得到正方形尺寸
    return img, (h0, w0), (new_h, new_w), (dw, dh)


if __name__ == '__main__':
    path1 = '../data_info/my_val_data.txt'
    datasets = LoadImagesAndLabels(path=path1)
    print(len(datasets))
