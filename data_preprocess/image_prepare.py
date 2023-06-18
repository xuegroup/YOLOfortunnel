# coding utf-8
# 作者：贾非
# 时间：2023/3/7 14:35
import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil


def cut(file_path, img_name, h_num, w_num, save_path):
    """
    :param file_path: 图像所在文件夹路径
    :param img_name: 图像名称
    :param h_num: 高度方向裁剪数目
    :param w_num: 宽度方向裁剪数目
    :param save_path: 切割后图像保存路径
    :return: 无
    """
    img_array = cv2.imread(os.path.join(file_path, img_name), cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img_array.shape[0], img_array.shape[1]
    cut_h, cut_w = int(img_h / h_num), int(img_w / w_num)

    h_id = 0
    for i in range(h_num):
        w_id = 0
        for j in range(w_num):
            im = img_array[h_id:h_id + cut_h, w_id:w_id + cut_w]
            # 按照裁剪图像在原图位置进行命名，行列(hw)
            cv2.imwrite(os.path.join(save_path, img_name.split('.')[0] + '_{}{}.png'.format(i, j)), im)
            w_id += cut_w
        h_id += cut_h


def concat(img_path, save_path, h_num):
    """
    :param img_path: 待拼接图像所在路径
    :param save_path: 拼接后图像保存路径
    :param h_num: 原图在高度方向裁剪数目
    :return: 无
    """
    image_list = []
    for image in os.listdir(img_path):
        im = image.split('_')
        im.pop(-1)
        if im not in image_list:
            image_list.append(im)

    for img_id in tqdm(image_list):
        concat_list = []
        for image in os.listdir(img_path):
            if all(i in image for i in img_id):
                concat_list.append(image)

        # 先水平方向拼（h方向），再竖直方向拼（v方向）
        im_v_list = []
        for idx in range(h_num):
            im_h_list = []
            for pic in concat_list:
                i, j = pic.split('_')[-1].split('.')[0]  # 对应上面命名中添加的行列名称
                if int(i) == idx:
                    im_array = cv2.imread(os.path.join(img_path, pic), cv2.IMREAD_GRAYSCALE)
                    im_h_list.append(im_array)
            im_v_list.append(np.concatenate(im_h_list, axis=1))
        im_final = np.concatenate(im_v_list, axis=0)
        im_name = '_'.join(img_id)
        cv2.imwrite(os.path.join(save_path, im_name + '.png'), im_final)


def image_rotate(file_path, save_path):
    """
    :param file_path: 文件路劲
    :param save_path: 保存路径
    :return: 无
    上下行图像导出时渗水形态与常识不符，根据上下行对图像进行翻转，这里只是针对本数据集特殊情况，如有需要可修改
    """
    for im in tqdm(os.listdir(file_path)):
        name_list = im.split('_')
        img = cv2.imread(os.path.join(file_path, im), cv2.IMREAD_GRAYSCALE)
        if name_list[0] == 's':
            img = cv2.flip(img, 0)  # 水平翻转
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(save_path, im), img)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(os.path.join(save_path, im), img)


def move_image(label_path):
    """
    :param label_path: 多人标注标签在的总文件夹
    :return: 无
    多人标注的图像在images文件夹中，标注文件在labels_xml文件夹中，分别以姓名命名
    根据标签文件名称将图像复制至标签文件夹
    """
    image_path = label_path.replace('labels_xml', 'images')
    for labeler in os.listdir(label_path):
        print('reading {} labels'.format(labeler))
        for label in os.listdir(os.path.join(label_path, labeler)):
            img_name = label.split('.')[0] + '.png'
            shutil.copy(os.path.join(image_path, labeler, img_name), os.path.join(label_path, labeler, img_name))


# 伽马变换，通过查表方式获得，cv2.LUT
def gamma_trans(im, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(im, gamma_table), gamma_table


if __name__ == '__main__':

    # file_path1 = r'C:\JiaFeiProjects\my_yolo_datasets_rotate'  # 绝对路径，路径中不能有中文
    # save_path1 = r'C:\JiaFeiProjects\my_yolo_datasets_cut'
    # for f in tqdm(os.listdir(file_path1)):
    #     cut(file_path=file_path1, img_name=f, h_num=4, w_num=2, save_path=save_path1)

    # file_path2 = r'E:\JiaFeiProjects\test_labelimg_xml\cut_rename'
    # save_path2 = r'E:\JiaFeiProjects\test_labelimg_xml\concat'
    # concat(img_path=file_path2, save_path=save_path2, h_num=4)

    # file_path3 = r'C:\JiaFeiProjects\my_yolo_datasets'  # 绝对路径，路径中不能有中文
    # save_path3 = r'C:\JiaFeiProjects\RAILWAY_images'
    # image_rotate(file_path=file_path3, save_path=save_path3)
    pass
