# -*- coding utf-8 -*-
# 作者：贾非
# 时间：2023/5/1 下午7:35
import json
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model.yolo import YOLO
from utils.loss_NMS import non_max_suppression
from utils.train_eval_utils import scale_box


# YOLOv5调色板，线条颜色，字体颜色等
class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))  # 转为16进制


class Annotator:
    def __init__(self, im):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input image'
        self.im = im
        self.lw = max(round(sum(im.shape) / 2 * 0.003), 2)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        h0, w0 = self.im.shape[:2]
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            font_thick = max(self.lw - 1, 1)
            # text width, height
            w, h = cv2.getTextSize(label, fontFace=0, fontScale=self.lw / 3, thickness=font_thick)[0]
            h_outside = p1[1] - h >= 3
            w_outside = p1[0] + w >= w0
            font_p1 = p2[0] - w if w_outside else p1[0], p1[1]
            font_p2 = p2[0] if w_outside else font_p1[0] + w, font_p1[1] - h - 3 if h_outside else font_p1[1] + h + 3
            cv2.rectangle(self.im, font_p1, font_p2, color, thickness=-1, lineType=cv2.LINE_AA)  # filled rectangle
            cv2.putText(self.im, label, (font_p1[0], font_p1[1] - 2 if h_outside else font_p1[1] + h + 2),
                        fontFace=0, fontScale=self.lw / 3, color=txt_color, thickness=font_thick, lineType=cv2.LINE_AA)

    def save_res(self, path):
        return cv2.imwrite(path, self.im)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# one image inference one time
def main():
    cfg = './cfg/yolov5l.yaml'
    data = './data_info/my_train_data.txt'
    weights = './20230428/yolo-99.pt'
    classes_json = './data_info/my_classes.json'
    save_path = './predict_res'
    img_size = 640

    # data path
    with open(data, 'r') as da:
        f = da.read().splitlines()
    img_files = [x for x in f]

    # label num for name
    num2name = {}
    with open(classes_json, 'r') as cls:
        cls_dict = json.load(cls)
    for i, name in enumerate(cls_dict):
        num2name[i] = name

    # create model and load weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(cfg=cfg, ch=1)
    weights_dict = torch.load(weights, map_location='cpu')
    weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    # start inference
    with torch.no_grad():
        # init
        img = torch.zeros((1, 1, img_size, img_size), device=device)
        model(img)

        colors = Colors()
        pbar = tqdm(img_files)
        forward_time, NMS_time, inference_time = 0, 0, 0
        for i, im_path in enumerate(pbar):
            pbar.set_description(f'inference [{i}]/[{len(img_files)}] images')
            # 和my_datasets中对图像处理相似，load_image()
            img0 = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            annotator = Annotator(img0)  # 实例化可视化类
            h0, w0 = img0.shape[:2]
            r = img_size / max(h0, w0)
            new_w, new_h = int(w0 * r), int(h0 * r)
            img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_AREA)
            dw, dh = img_size - new_w, img_size - new_h
            img = cv2.copyMakeBorder(img, top=0, bottom=dh, left=0, right=dw,
                                     borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
            img = img[..., None].transpose(2, 0, 1)  # add channel dimension
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)  # add batch dimension
            # image read and process end，数据加载代码结束

            # forward
            t1 = time_synchronized()
            pred = model(img)[0]
            t2 = time_synchronized()

            # NMS
            pred = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.4)[0]
            t3 = time_synchronized()

            if pred is None:
                print(f'No targets detect {im_path}')
                continue
            # print(f'forward time: {t2 - t1}, NMS time: {t3 - t2}, inference time: {t3 - t1}')
            forward_time += t2 - t1
            NMS_time += t3 - t2
            inference_time += t3 - t1
            pbar.set_postfix({'forward time': forward_time / (i + 1),
                              'NMS time': NMS_time / (i + 1),
                              'inference time': inference_time / (i + 1)})

            pred[:, :4] = scale_box((img_size, img_size), pred[:, :4], (h0, w0)).round()
            boxes = pred[:, :4].detach().cpu().tolist()
            scores = pred[:, 4].detach().cpu().float()
            classes = pred[:, 5].detach().cpu().float()
            for box, score, cls in zip(boxes, scores, classes):
                labels = f'{num2name[int(cls)]}: {round(float(score), 3)}'
                annotator.box_label(box, labels, color=colors(int(cls), True))
            im_name = im_path.split('/')[-1]
            annotator.save_res(os.path.join(save_path, im_name))


if __name__ == '__main__':
    main()
