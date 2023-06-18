# -*- coding utf-8 -*-
# 作者：贾非
# 时间：2023/4/21 下午4:25
from pycocotools.coco import COCO
from tqdm import tqdm


# 将验证数据集转为COCO可读取的格式
def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1  # 标签应该从1开始
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()  # 创建一个无序、不重复的集合set对象
    # 遍历dataset中的每张图像
    for img_idx in tqdm(range(len(ds)), desc='loading eval info for coco tools'):
        # targets [num_obj, 5] 5 means (cls, x, y, w, h) tensor type
        targets, shapes = ds.coco_index(img_idx)
        img_dict = {}
        img_dict['id'] = img_idx
        img_dict['height'] = shapes[0]
        img_dict['width'] = shapes[1]
        dataset['images'].append(img_dict)

        for obj in targets:
            ann = {}
            ann['image_id'] = img_idx
            boxes = obj[1:]
            # 相对转为绝对，coco框坐标为左上角 （xmin, ymin, w, h）
            boxes[:2] -= 0.5 * boxes[2:]
            boxes[[0, 2]] *= img_dict['width']
            boxes[[1, 3]] *= img_dict['height']
            boxes = boxes.tolist()

            ann['bbox'] = boxes
            ann['category_id'] = int(obj[0])
            categories.add(int(obj[0]))
            ann['area'] = boxes[2] * boxes[3]
            ann['iscrowd'] = 0
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
