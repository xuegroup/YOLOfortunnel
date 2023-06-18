# coding utf-8
# 作者：贾非
# 时间：2023/3/22 12:47
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils.distributed_utils import de_parallel


def smooth_BCE(eps=0.1):
    # 让真实标签不是绝对的1，背景不是绝对的0
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLoss:

    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        m = de_parallel(model).model[-1]  # Detect Module
        # loss平衡系数，增大预测小目标特征图的权重，后面get对应的是1280大尺寸网络时
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.BCEcls, self.BCEobj, self.hyp = BCEcls, BCEobj, h
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors  # (nl, na, 2)
        self.device = device

    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # obj loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            nb = b.shape[0]  # 正样本数量
            if nb:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), dim=1)
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), dim=1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()  # 正样本的回归损失

                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = iou  # 计算所有样本的obj_loss，为正样本设置标签，即iou

                if self.nc > 1:
                    # 手动做了one-hot，将对应位置标签设为1
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(nb), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # loss scaled by batch_size

        return (lbox + lobj + lcls) * bs, {'box_loss': lbox, 'obj_loss': lobj, 'cls_loss': lcls}

    def build_targets(self, p, targets):
        # targets (img_idx, cls, x, y, w, h) shape (nt, 6)
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)

        # 为target添加anchor索引，利用mask匹配anchor和target
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # (na, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), dim=2)  # (na, nt, 6+1)

        g = 0.5
        # 增加匹配到的正样本数量
        # 首先先全都保存，之后再扩展 j, k, l, m (5, 2), 向左、向上、向右、向下
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            # shape: [bs, na, h, w, no]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain  # (na, nt, 7) 转到当前特征图的绝对坐标
            if nt:
                r = t[..., 4:6] / anchors[:, None]  # (na, nt, 2) wh_ratio
                j = torch.max(r, 1 / r).max(dim=2)[0] < self.hyp['anchor_t']  # compare (na, nt)
                t = t[j]  # (N, 7) 匹配成功的targets及其anchor索引

                # 计算匹配成功的是否可以继续匹配上下左右的网格点
                gxy = t[:, 2:4]  # (N, 2)
                gxi = gain[[2, 3]] - gxy  # (N, 2)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # (N) (N)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # (N) (N)
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # (5, N)
                t = t.repeat((5, 1, 1))[j]  # (5, N, 7) -> (N', 7)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # (5, N, 2) -> (N', 2)
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, dim=1)  # (img_idx, cls) (x, y) (w, h) anchor_idx
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()  # 所属网格左上角坐标
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), dim=1))  # 模型需要预测的坐标偏移真实值与宽高真实值
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 两矩形外接矩形的宽
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 两矩形外接矩形的高
        if CIoU or DIoU:
            diagonal2 = cw ** 2 + ch ** 2 + eps  # 两矩形外接矩形的对角线长度
            # 两矩形中心点距离
            center2 = ((b2_x2 + b2_x1 - b1_x2 - b1_x1) ** 2 + (b2_y2 + b2_y1 - b1_y2 - b1_y1) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + eps)
                return iou - (center2 / diagonal2 + v * alpha)  # CIoU
            return iou - center2 / diagonal2  # DIoU
        c_area = cw * ch + eps  # 两矩形外接矩形的面积
        return iou - (c_area - union) / c_area  # GIoU
    return iou


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # xmin
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # ymin
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # xmax
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # ymax
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, agnostic=False):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # in eval mode, yolov5 output (torch.cat(z, dim=1), x) (bs, chn, no)

    device = prediction.device
    bs = prediction.shape[0]
    xc = prediction[..., 4] > conf_thres  # (bs, chn)

    max_wh = 7680  # add box offset for batch nms
    max_nms = 30000  # 进行nms的最大预测框数量
    time_limit = 0.5 + 0.05 * bs

    t = time.time()
    output = [torch.zeros((0, 6), device=device)] * bs
    for idx, x in enumerate(prediction):  # x (chn, no)
        x = x[xc[idx]]  # (N, no)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]  # conf = obj * cls
        box = xywh2xyxy(x[..., :4])  # (N, 4)

        conf, j = x[:, 5:].max(dim=1, keepdim=True)   # (N') (N')
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thres]  # (N', 4+1+1)

        n = x.shape[0]
        if not n:
            continue
        index = x[:, 4].argsort(descending=True)[:max_nms]
        x = x[index]

        # batch nms
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes scaled by offsets, scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[idx] = x[i]
        if (time.time() - t) > time_limit:
            print('NMS time exceed {:.3f}'.format(time_limit))
            break
    return output
