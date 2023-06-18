# coding utf-8
# 作者：贾非
# 时间：2023/3/16 16:06
import contextlib
import math
from torch.utils.tensorboard import SummaryWriter
import yaml

from model.layers import *


class Detect(nn.Module):
    stride = None  # compute when build model

    # 80 means COCO datasets classes
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of predicted feature map
        self.na = len(anchors[0]) // 2  # number of anchors

        # grid用于计算xy坐标，anchor_grid是用来计算wh宽高
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid (w, h)

        # 采用这种方式可以让模型也保存anchors参数
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # (nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # purely conv

    def forward(self, x):
        z = []  # for inference
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            # (bs, 255, h, w) -> (bs, 3, 85, h, w) -> (bs, 3, h, w, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # 将结果映射回训练图像尺寸大小，用于评估
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx=nx, ny=ny, i=i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), dim=4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                out = torch.cat((xy, wh, conf), dim=4)
                z.append(out.view(bs, self.na * nx * ny, self.no))

        # return x
        return x if self.training else (torch.cat(z, dim=1), x)

    # 以输入640为例，对第一个特征层，缩放32倍，即为20，该值在计算中会随模型输出改变
    # 只有在推理时才会生成，用于反算坐标，进行评估
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        # 生成特征图网格
        grid = torch.stack((xv, yv), dim=2).expand(shape) - 0.5  # add grid offset i.e., y = 2.0 * sigma(x) - 0.5
        # 将anchor放到每一个grid point
        # 在模型搭建时，anchors按照stride缩放至相对坐标，所以这里要再反算回去
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)

        return grid, anchor_grid


def check_anchor_order(m):
    a = m.anchors.prod(-1).mean(-1).view(-1)  # 每种规模预测层上anchor面积均值
    a_m = a[-1] - a[0]
    s_m = m.stride[-1] - m.stride[0]
    if a_m and (a_m.sign() != s_m.sign()):
        print('Reversing anchor orders')
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # 默认为nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.ReLU, nn.LeakyReLU, nn.SiLU]:
            m.inplace = True


class YOLO(nn.Module):
    # model_yaml, in_image_channel
    def __init__(self, cfg='../cfg/yolov5l.yaml', ch=3):
        super().__init__()
        with open(cfg) as f:
            self.yaml = yaml.safe_load(f)  # model dict

        # define model
        self.model, self.save = parse_model(self.yaml, [ch])  # model, save_list
        self.nc = self.yaml['nc']
        self.names = [str(i) for i in range(self.yaml['nc'])]

        # compute stride for detect head
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256  # 为了计算输出特征图尺寸设置的一个可以被32整除的数
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)  # 确保anchor与stride顺序对上
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # 手动对Detect偏置进行初始化

        initialize_weights(self)  # 手动初始化模型整体参数权重

    def forward(self, x):
        return self._forward_once(x)

    def _forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:  # from earlier layers
                x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)  # 为了确保索引与yaml中对的上，这里全部都保存，只不过不满足条件的置为None
        return x

    def _initialize_biases(self):
        m = self.model[-1]  # Detect
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)  # 255 -> (3, 85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):  # yaml_dict, in_channel_list
    # 函数的目的除了把模型框架搭起来
    # cfg中args存的是当前层的输出维度，ch的作用就是把这个维度存起来，作为下一层的输入
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)
        print(f'WARNING default activation changed to {act}')
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)  # number of outputs per anchor i.e.,(x, y, w, h, conf, nc) * na

    # layers就是构建的模型，用Sequential组成
    # save是指明哪些层的输出要保存，比如Concat要拼接几层的输出，也就是from中大于0的数字
    # 在模型计算时，根据idx得到该层的输出
    layers, save, c2 = [], [], ch[-1]
    for idx, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # 定义在layer中的module

        for jdx, a in enumerate(args):
            with contextlib.suppress(NameError):  # 一些默认参数直接返回None，避免报错，如"nearest"
                args[jdx] = eval(a) if isinstance(a, str) else a

        n = max(round(n * gd), 1) if n > 1 else n  # 不同版本模型搭建需求，指的是Bottleneck重复次数
        if m in {Conv, SPPF, C3, Bottleneck}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 不是输出层的话，对不同版本需要改变输出维度，模型浅输出维度不能过大
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)  # Bottleneck重复次数
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]  # 输出维度与上一层一致

        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())
        # m.i的作用就是在forward时，与save中索引对应，保存那一层的输出
        m_.i, m_.f, m_.type, m_.np = idx, f, t, np  # index, from_index, module_type, num_params

        save.extend(x % idx for x in ([f] if isinstance(f, int) else f) if x != -1)  # 就是把f中大于0的取出来，操作略复杂
        layers.append(m_)
        # 将输入图像维度去除，索引0从第一个特征层的输出维度开始，作为下一层的输入维度
        if idx == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # tensorboard画的......略微有些复杂，有数画不出来。。。
    mm = SummaryWriter()
    aa = YOLO()
    test_input = torch.zeros([1, 3, 640, 640])
    mm.add_graph(aa, test_input, use_strict_trace=False)
