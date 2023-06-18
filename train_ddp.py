# coding utf-8
# 作者：贾非
# 时间：2023/3/30 19:03
import argparse
import datetime
import math
import os
import time

import torch
import yaml
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from model.yolo import YOLO
from utils.distributed_utils import init_distributed_mode, torch_distributed_zero_first
from dataset.my_datasets import LoadImagesAndLabels
from utils.train_eval_utils import train_one_epoch, evaluate


def main(opt, hyp):
    init_distributed_mode(opt)

    if opt.rank in [-1, 0]:
        print(opt)
        print('create tensorboard')
        tb_writer = SummaryWriter()

    device = torch.device(opt.device)
    if 'cuda' not in device.type:
        raise EnvironmentError('no find gpu for training')
    cfg = opt.cfg
    train_data = opt.train_data
    val_data = opt.val_data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / (opt.world_size * opt.batch_size)), 1)  # 累计数个batch梯度再更新
    imgsz_train = opt.img_size
    imgsz_val = opt.img_size

    result_file = 'result{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    # dataset
    with torch_distributed_zero_first(opt.rank):
        train_dataset = LoadImagesAndLabels(train_data, imgsz_train, rank=opt.rank)
        val_dataset = LoadImagesAndLabels(val_data, imgsz_val, rank=opt.rank)

    train_sampler = data.distributed.DistributedSampler(train_dataset)
    val_sampler = data.distributed.DistributedSampler(val_dataset)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=nw,
                                        pin_memory=True, collate_fn=train_dataset.collate_fn, drop_last=True)
    val_data_loader = data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=nw,
                                      pin_memory=True, collate_fn=val_dataset.collate_fn, drop_last=True)

    # creating model
    start_epoch = 0
    best_mAP = 0.0
    model = YOLO(cfg=cfg, ch=1).to(device)
    nc = model.nc
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp['momentum'],
                          weight_decay=hyp['weight_decay'], nesterov=True)

    # scheduler
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 从零开始就是0，如何断点续训需要设置

    # some extra properties
    hyp['cls'] *= nc / 80
    model.hyp = hyp

    # start training
    if opt.rank in [-1, 0]:
        print(f'start training for {epochs}')
        print(f'using {nw} dataloader workers')
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        mloss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                    world_size=opt.world_size, print_freq=50, accumulate=accumulate,
                                    warmup=True)

        # update scheduler
        scheduler.step()

        # 每epoch评估一次模型，保存训练结果、模型权重
        if opt.no_test is False or epoch == epochs - 1:
            result_info = evaluate(model, val_data_loader, device=device)
            if opt.rank in [-1, 0]:
                coco_mAP = result_info['bbox'][0]
                voc_mAP = result_info['bbox'][1]
                coco_mAR = result_info['bbox'][8]

                # tensorboard显示
                if tb_writer:
                    tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', 'learning_rate',
                            'coco/mAP@[IoU=0.50:0.95]', 'coco/mAP[IoU=0.50]', 'coco/mAR[IoU=0.50:0.95]']
                    for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # 结果写入txt文件
                with open(result_file, 'a') as f:
                    # save as csv for plot with windows excel
                    if epoch == 0:
                        f.write(','.join(('epoch', 'mAP', 'AP.5', 'AP.75', 'APs', 'APm', 'APl',
                                          'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl',
                                          'box_loss', 'obj_loss', 'cls_loss', 'loss', 'lr_rate')) + '\n')
                    final_list = [str(epoch)] + [str(round(i, 4)) for i in result_info['bbox']] + \
                                 [str(round(i, 4)) for i in mloss.tolist()] + [str(round(lr, 6))]
                    f.write(','.join(final_list) + '\n')

                if coco_mAP > best_mAP:
                    best_mAP = coco_mAP

                # 存储模型
                with open(result_file, 'r') as f:
                    save_dict = {'model': model.module.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'training_results': f.read(),
                                 'epoch': epoch,
                                 'best_mAP': best_mAP}
                    torch.save(save_dict, f'./weights/yolo-{epoch}.pt')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    if opt.rank in [-1, 0]:
        print(f'training total time {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='./data_info/my_train_data.txt',
                        help='train data txt path')
    parser.add_argument('--val-data', type=str, default='./data_info/my_val_data.txt',
                        help='val data txt path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=3, help='images per gpu')
    parser.add_argument('--no-test', action='store_true', help='only test final epoch')
    parser.add_argument('--cfg', default='./cfg/yolov5l.yaml', help='model config path')
    parser.add_argument('--hyp', default='./cfg/hyp.no-augmentation.yaml', help='hyperparameter path')
    parser.add_argument('--img-size', type=int, default=640, help='test size')
    parser.add_argument('--weights', type=str, default=None, help='checkpoint path / pretrained weights')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='freeze backbone or some else layers')

    parser.add_argument('--device', default='cuda', help='device id')
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url to set up distributed training')

    opt = parser.parse_args()

    with open(opt.hyp, 'r') as f:
        hyp = yaml.safe_load(f)
    print(hyp)

    main(opt, hyp)
