# coding utf-8
# 作者：贾非
# 时间：2023/3/7 20:32
import os
import json
import copy
from lxml import etree
from tqdm import tqdm

from data_preprocess import parse_xml_to_dict


def check_label(root_path, name='train'):
    """
    :param name:
    :param root_path: 标签文件夹
    :return: 无
    """
    class_name = ['crack', 'leak', 'block']
    label_path = os.path.join(root_path, name, 'labels_xml')
    labels = os.listdir(label_path)
    img_path = os.path.join(root_path, name, 'images')
    images = [im.split('.')[0] for im in os.listdir(img_path)]
    assert len(images) == len(labels), 'unmatched images and labels'

    for label in tqdm(labels):
        if label.split('.')[0] not in images:
            print(f'check {label} in images')

        with open(os.path.join(label_path, label), 'r', errors='ignore') as f:
            xml_str = f.read()
        xml_doc = etree.fromstring(xml_str)
        xml_dict = parse_xml_to_dict(xml_doc)
        if 'object' not in xml_dict['annotation']:
            print('{} no object'.format(os.path.join(label_path, label)))
            # shutil.move(os.path.join(label_path, label),
            #             os.path.join(r'C:\JiaFeiProjects\invalid', label))
        else:
            for obj in xml_dict['annotation']['object']:
                xmin = float(obj['bndbox']['xmin'])
                xmax = float(obj['bndbox']['xmax'])
                ymin = float(obj['bndbox']['ymin'])
                ymax = float(obj['bndbox']['ymax'])

                if xmax <= xmin or ymax <= ymin:
                    print(f'{os.path.join(label_path, label)} wrong label format')
                if obj['name'] not in class_name:
                    print('{} invalid label'.format(os.path.join(label_path, label)))


def label_cal(root, classes_json, name=None):
    """
    :param root: 训练集与验证集图像，标签文件等所在总文件夹
    :param classes_json: 类别名称json文件
    :param name: 训练集与验证集文件夹名称
    :return: 无
    """
    if name is None:
        name = ['train']
    with open(classes_json, 'r') as cls:
        label_dict = json.load(cls)

    data_dict = {}
    for n in name:
        ld_n = copy.copy(label_dict)
        for key in ld_n:
            ld_n[key] = 0

        path = os.path.join(root, n, 'labels_xml')
        for label in tqdm(os.listdir(path), desc='calculating labels'):
            with open(os.path.join(path, label), 'r', errors='ignore') as f:
                xml_string = f.read()
            xmla = etree.fromstring(xml_string)
            xml_dic = parse_xml_to_dict(xmla)
            for i in xml_dic['annotation']['object']:
                assert i['name'] in ld_n, '{} file wrong label'.format(label)
                ld_n[i['name']] += 1
        data_dict[n] = ld_n
    print(data_dict)


def create_json(label_name):
    """
    :param label_name: 一个列表，用户自己写一个传入，类似[crack, leak, block]
    :return: 无
    """
    label_dict = {}
    for i, l in enumerate(label_name):
        label_dict[l] = i
    with open('../data_info/my_classes.json', 'w') as cls_f:
        json.dump(label_dict, cls_f, indent=0)


if __name__ == '__main__':

    # a = ['crack', 'leak', 'block']
    # create_json(a)

    root1 = '/home/ubantu/jiafei_projects/yolo_dataset'
    classes_json1 = '../data_info/my_classes.json'
    label_cal(root1, classes_json1, ['train', 'val'])

    # root_path1 = '/home/ubantu/jiafei_projects/yolo_dataset'
    # check_label(root_path=root_path1)
