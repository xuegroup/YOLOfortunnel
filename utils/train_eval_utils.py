# coding utf-8
# 作者：贾非
# 时间：2023/3/21 17:43
import sys

from utils.coco_eval import CocoEvaluator
from utils.coco_utils import convert_to_coco_api
from utils.loss_NMS import *
from utils.train_process_display import *
from utils.distributed_utils import *


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, accumulate, world_size=1, warmup=False):
    model.train()
    compute_loss = ComputeLoss(model)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=warmup_factor,
                                                         total_iters=warmup_iters)
        accumulate = 1

    mloss = torch.zeros(4).to(device)
    now_lr = 0.
    nb = len(data_loader)
    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = i + nb * epoch  # 从训练开始到目前为止总共经历了多少个batch
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        pred = model(imgs)
        losses, loss_dict = compute_loss(pred, targets)  # loss scaled by bs
        losses *= world_size  # gradients averaged by GPU nums in DDP mode

        # for log purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_items = torch.cat((loss_dict_reduced['box_loss'],
                                loss_dict_reduced['obj_loss'],
                                loss_dict_reduced['cls_loss'],
                                losses_reduced)).detach()
        mloss = (mloss * i + loss_items) / (i + 1)

        if not torch.isfinite(losses_reduced):
            print('WARNING: non-finite loss, end training', loss_dict_reduced)
            print('training images path {}'.format(','.join(paths)))
            sys.exit(1)
        losses.backward()

        # 每accumulate个bacth更新一个参数
        if ni % accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:
            lr_scheduler.step()

    return mloss, now_lr


# 将训练尺寸上预测框坐标反算至原始图像，对应于dataset中数据加载时图像处理方法的逆向运算
def scale_box(train_img_shape, boxes, ori_img_shape):
    ratio = max(ori_img_shape) / max(train_img_shape)
    boxes[:, [0, 2]] *= ratio
    boxes[:, [1, 3]] *= ratio

    boxes[:, 0].clamp_(0, ori_img_shape[1])
    boxes[:, 1].clamp_(0, ori_img_shape[0])
    boxes[:, 2].clamp_(0, ori_img_shape[1])
    boxes[:, 3].clamp_(0, ori_img_shape[0])
    return boxes


@torch.no_grad()
def evaluate(model, data_loader, iou_types: list = None, coco=None, device=None):
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Test: '

    if coco is None:
        coco = convert_to_coco_api(data_loader.dataset)
    if iou_types is None:
        iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0

        if device != cpu_device:
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred = model(imgs)[0]
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6)
        model_time = time.time() - model_time

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

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = {}
    for iou_type in iou_types:
        result_info[iou_type] = coco_evaluator.coco_eval[iou_type].stats.tolist()

    return result_info
