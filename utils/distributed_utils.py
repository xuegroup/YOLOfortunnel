# coding utf-8
# 作者：贾非
# 时间：2023/3/21 17:56
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    return model.module if is_parallel(model) else model


def all_gather(data):
    # 主要是对模型推理结果进行处理，将所有进程的预测结果进行汇总
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    dist_list = [None] * world_size
    dist.all_gather_object(dist_list, data)  # all_gather只能接受tensor，模型预测结果一般是python字典，故用这种形式
    return dist_list


def reduce_dict(input_dict, average=True):
    # 主要是对loss进行处理，用于展示
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    # 对于多机多卡，world_size一般指机器数量，rank指所有机器gpu的排号,local_rank指限于一台机器时，gpu的排号
    # 对于单机多卡，world_size指gpu数量，此时rank和local_rank代表的就是一样的含义
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    # 对于所有进程定义了一个print函数代替了内置的print函数，保证只在主进程输出
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank * torch.cuda.device_count()
    else:
        print('Unable to use distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


@contextmanager
def torch_distributed_zero_first(local_rank):
    # 确保所有进程在进行同一过程时，每个进程都完成了，再进行下一步操作
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()
