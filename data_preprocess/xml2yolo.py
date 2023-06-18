# coding utf-8
# 作者：贾非
# 时间：2023/3/7 21:05
from lxml import etree
from tqdm import tqdm
import os
import json

from data_preprocess import parse_xml_to_dict


def trans_write_label_info(root, classes_json, train_val='train'):
    """
    :param root: 数据集所在文件夹
    :param classes_json: create_json(data_analysis)生成的类别文件
    :param train_val: 训练集还是测试集
    :return: 无
    """
    with open(classes_json, 'r') as cls:
        class_dict = json.load(cls)

    path = os.path.join(root, train_val)
    label_path = os.path.join(path, 'labels_xml')
    for name in tqdm(os.listdir(label_path), desc=f'translating {train_val} info'):
        with open(os.path.join(label_path, name), 'r', errors='ignore') as file:
            xml_str = file.read()
        xml = etree.fromstring(xml_str)
        xml_dict = parse_xml_to_dict(xml)
        img_w = int(xml_dict['annotation']['size']['width'])
        img_h = int(xml_dict['annotation']['size']['height'])

        with open(os.path.join(path, 'labels', name.split('.')[0]+'.txt'), 'w') as la:
            for obj in xml_dict['annotation']['object']:
                xmin = float(obj['bndbox']['xmin'])
                xmax = float(obj['bndbox']['xmax'])
                ymin = float(obj['bndbox']['ymin'])
                ymax = float(obj['bndbox']['ymax'])
                class_idx = class_dict[obj['name']] - 1  # 类别id从0开始

                xc = xmin + (xmax - xmin) / 2
                yc = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                xc = round(xc / img_w, 6)
                yc = round(yc / img_h, 6)
                w = round(w / img_w, 6)
                h = round(h / img_h, 6)

                info = [str(i) for i in [class_idx, xc, yc, w, h]]
                la.write(' '.join(info) + '\n')


# 分别生成训练集与验证集图像的绝对路径
def create_data_txt(root, train_val='train'):
    img_path = os.path.join(root, train_val, 'images')
    with open('../data_info/my_{}_data.txt'.format(train_val), 'w') as f:
        for im in os.listdir(img_path):
            f.write(os.path.join(img_path, im) + '\n')


if __name__ == '__main__':
    # root1 = '/home/ubantu/jiafei_projects/yolo_dataset'
    # classes_json1 = '../data_info/my_classes.json'
    # trans_write_label_info(root=root1, classes_json=classes_json1, train_val='val')

    root2 = '/home/ubantu/jiafei_projects/yolo_dataset'
    create_data_txt(root=root2, train_val='val')
