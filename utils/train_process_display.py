# coding utf-8
# 作者：贾非
# 时间：2023/3/21 17:55
import datetime
import time

import torch
import torch.distributed as dist
from collections import defaultdict, deque
from utils.distributed_utils import is_dist_avail_and_initialized


class SmoothValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f}({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg,
                               global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(SmoothValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("{} object has no attribute {}".format(type(self).__name__, attr))

    def __str__(self):
        print_str = []
        for name, value in self.meters.items():
            print_str.append('{}: {}'.format(name, str(value)))
        return self.delimiter.join(print_str)

    def synchronize_between_processes(self):
        for value in self.meters.values():
            value.synchronize_between_processes()

    def add_meter(self, name, value):
        self.meters[name] = value

    def log_every(self, iterable, print_freq, header=None):
        if header is None:
            header = ''
        start_time = time.time()  # 训练开始时间
        end = time.time()  # 每个batch结束时间，不断增加，直到最后训练结束时间
        iter_time = SmoothValue(fmt='{avg:.4f}')  # 每个batch计算时间
        data_time = SmoothValue(fmt='{avg:.4f}')  # 每个batch加载时间
        # [0 /[35]表示当前epoch/总epoch，0之后间距为了美观设置一个间距
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        i = 0
        MB = 1024.0 * 1024.0
        # 开始一个epoch训练
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                # Estimated Time of Arrival，预计到达时间
                # 根据当前训练每次iter的全局平均时间，估计这次epoch的结束还要多长时间
                # datetime.timedelta可以进行转换，例传入eta_seconds=3600，输出1:00:00即1小时
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i, len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f}) s / iter'
              .format(header, total_time_str, total_time / len(iterable)))
