# -*- coding utf-8 -*-
# 作者：贾非
# 时间：2023/5/3 下午3:57
import os

import torch
import torch.utils.data as data
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.my_datasets import LoadImagesAndLabels
from model.yolo import YOLO
from utils.coco_utils import convert_to_coco_api
from utils.coco_eval import convert_to_coco_xywh
from utils.loss_NMS import non_max_suppression
from utils.train_eval_utils import scale_box


# single gpu evaluate
@torch.no_grad()
def validate_coco():
    cfg = './cfg/yolov5l.yaml'
    val_data = './data_info/my_val_data.txt'
    weights = './20230503/yolo-99.pt'
    img_size = 640
    batch_size = 1
    cpu_device = torch.device('cpu')

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_dataset = LoadImagesAndLabels(val_data, img_size)
    val_dataloader = data.DataLoader(val_dataset, batch_size, num_workers=nw,
                                     collate_fn=val_dataset.collate_fn, pin_memory=True)

    # create model and load weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO(cfg=cfg, ch=1)
    weights_dict = torch.load(weights, map_location='cpu')
    weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    # start validate
    coco_results = []
    pbar = tqdm(val_dataloader)
    for imgs, targets, paths, shapes, img_index in pbar:
        imgs = imgs.to(device).float() / 255.0

        if device != cpu_device:
            torch.cuda.synchronize(device)

        pred = model(imgs)[0]
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6)

        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                boxes = p[:, :4]
                boxes = scale_box(imgs[index].shape[1:], boxes, shapes[index]).round()

            info = {'boxes': boxes.to(cpu_device),
                    'labels': p[:, 5].to(cpu_device, dtype=torch.int64),
                    'scores': p[:, 4].to(cpu_device)}
            outputs.append(info)

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        # save as coco_dt format
        for original_id, prediction in res.items():
            if len(prediction) == 0:
                continue

            boxes = prediction['boxes']
            boxes = convert_to_coco_xywh(boxes).tolist()
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            coco_results.extend([
                {
                    'image_id': original_id,
                    'category_id': labels[k],
                    'bbox': box,
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
            ])

    coco_gt = convert_to_coco_api(val_dataloader.dataset)
    coco_dt = COCO.loadRes(coco_gt, coco_results)
    coco_evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == '__main__':
    validate_coco()
