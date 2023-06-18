# coding utf-8
# 作者：贾非
# 时间：2023/3/9 15:31
import os
import cv2
import shutil
from lxml import etree
from tqdm import tqdm

from data_preprocess import parse_xml_to_dict


# 切割后图像中检测标签反算至原图，再按照一定规则进行调整，提取正方形图像形成分类数据集
def extract_box(root, train_val='train', cut_num=2048):

    name_list = open(os.path.join(root, train_val, train_val+'.txt'), 'r').read().splitlines()
    for idx, n in enumerate(tqdm(name_list, desc='extracting bbox')):
        a = n.split('_')
        i, j = a[-1]
        i, j = int(i), int(j)
        ori_im = '_'.join(a[:3])
        img = cv2.imread(os.path.join(root, 'my_yolo_datasets_rotate', ori_im+'.png'), cv2.IMREAD_GRAYSCALE)

        with open(os.path.join(root, train_val, 'labels_xml', n+'.xml'), 'r', errors='ignore') as f:
            xml_str = f.read()
            xml_doc = etree.fromstring(xml_str)
            xml_dic = parse_xml_to_dict(xml_doc)

        for jdx, obj in enumerate(xml_dic['annotation']['object']):
            xmin = int(obj['bndbox']['xmin'])
            xmax = int(obj['bndbox']['xmax'])
            ymin = int(obj['bndbox']['ymin'])
            ymax = int(obj['bndbox']['ymax'])
            cls_name = obj['name']
            w = xmax - xmin
            h = ymax - ymin

            # 原图坐标
            xmin += int(j)*cut_num
            xmax += int(j)*cut_num
            ymin += int(i)*cut_num
            ymax += int(i)*cut_num

            limit_length = int(cut_num * 0.8)
            if w > limit_length or h > limit_length:
                if cls_name != 'crack':
                    shutil.copy(os.path.join(root, train_val, 'images', n+'.png'),
                                os.path.join(root, train_val, cls_name, n+'.png'))
                else:
                    if w > 2 * h:
                        wm = int(w / 2)
                        xm = xmin + wm
                        ymin -= int((wm - h) / 2)
                        ymax += int((wm - h) / 2)
                        cv2.imwrite(os.path.join(root, train_val, cls_name, '{}{}_1.png'.format(idx, jdx)),
                                    img[ymin:ymax, xmin:xm])
                        cv2.imwrite(os.path.join(root, train_val, cls_name, '{}{}_2.png'.format(idx, jdx)),
                                    img[ymin:ymax, xm:xmax])
                    elif h > 2 * w:
                        hm = int(h / 2)
                        ym = ymin + hm
                        xmin -= int((hm - w) / 2)
                        xmax += int((hm - w) / 2)
                        cv2.imwrite(os.path.join(root, train_val, cls_name, '{}{}_1.png'.format(idx, jdx)),
                                    img[ymin:ym, xmin:xmax])
                        cv2.imwrite(os.path.join(root, train_val, cls_name, '{}{}_2.png'.format(idx, jdx)),
                                    img[ym:ymax, xmax:xmax])
                    else:
                        shutil.copy(os.path.join(root, train_val, 'images', n + '.png'),
                                    os.path.join(root, train_val, cls_name, n + '.png'))
                continue
            if w < 512 and h < 512:
                xmin -= int((512 - w) / 2)
                ymin -= int((512 - h) / 2)
                xmax += int((512 - w) / 2)
                ymin += int((512 - h) / 2)
            if w > h:
                if i == 0:
                    ymax += w - h
                elif i == 3:
                    ymin -= w - h
                else:
                    ymin -= int((w - h) / 2)
                    ymax += int((w - h) / 2)
            if w < h:
                if j == 0:
                    xmax += h - w
                else:
                    xmin -= h - w
            cv2.imwrite(os.path.join(root, train_val, cls_name, '{}{}.png'.format(idx, jdx)),
                        img[ymin:ymax, xmin:xmax])
