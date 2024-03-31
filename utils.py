'''
Code elements borrowed from 
https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
'''
import argparse
import os
import sys
from collections import defaultdict, deque
import time, datetime

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from einops import rearrange, reduce


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    args.distributed = True
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
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
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
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
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def multi_scale(samples, model, args):
    v = None
    for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = torch.nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)

        if args.pt_style == 'vicregl':
            feats = model(inp)[-1].clone()
        elif args.pt_style == 'clip':
            feats = model.module.encode_image(samples).to(torch.float32).clone()
        else:
            feats = model(inp).clone()
        feats = torch.squeeze(feats)
        feats = torch.unsqueeze(feats, 0)
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


def patchify(x, size):
    patches = rearrange(x, 'b c (h1 h2) (w1 w2) -> (b h1 w1) c h2 w2', h2=size, w2=size)
    return patches


@torch.no_grad()
def extract_features(args, model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    # count = 0
    for samples, index in metric_logger.log_every(data_loader, 100):
        print(f'At the index {index[0]}')
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = multi_scale(samples, model, args)
        else:

            if args.pt_style == 'dino':
                if args.layer > 1:
                    feats = model.module.get_intermediate_layers(samples, args.layer)[0][:, 0, :].clone()
                elif args.layer == -1:

                    allfeats = model.module.get_intermediate_layers(samples, len(model.module.blocks))
                    feats = [allfeats[i - 1][:, 0, :] for i in args.multilayer]
                    bdim, _ = feats[0].shape
                    feats = torch.stack(feats, dim=1).reshape((bdim, -1)).clone()
                else:
                    feats = model(samples).clone()

            elif args.pt_style == 'moco':
                feats = model.module.forward_features(samples)
                feats = feats[:, 0, :].clone()
            elif args.pt_style == 'vgg':
                feats = model.module.features(samples).clone()
            elif args.pt_style in ['clip', 'clip_wikiart']:
                #
                allfeats = model.module.visual.get_intermediate_layers(samples.type(model.module.dtype))
                # else:
                # allfeats = model.get_activations(samples) #[::-1]
                allfeats.reverse()

                if args.arch == 'resnet50':
                    # import ipdb; ipdb.set_trace()
                    if args.layer == -1:
                        raise Exception('Layer=-1 not allowed with clip resnet')
                    elif args.layer == 1:
                        feats = allfeats[0].clone()
                    else:
                        assert len(allfeats) >= args.layer, "Asking for features of layer that doesnt exist"
                        feats = reduce(allfeats[args.layer - 1], 'b c h w -> b c', 'mean').clone()

                else:
                    if args.layer == -1:
                        feats = [allfeats[i - 1][:, 0, :] for i in args.multilayer]
                        bdim, _ = feats[0].shape
                        feats = torch.stack(feats, dim=1).reshape((bdim, -1)).clone()
                    else:
                        assert len(allfeats) >= args.layer
                        feats = allfeats[args.layer - 1][:, 0, :].clone()
            else:
                feats = model(samples).clone()
        # init storage feature matrix
        feats = nn.functional.normalize(feats, dim=1, p=2).to(torch.float16)
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1], dtype=feats.dtype)
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l).cuda())
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())

    return features


def extract_features_pca(args, model, pca_model, k, data_loader, use_cuda=True, multiscale=False):
    metric_logger = MetricLogger(delimiter="  ")
    features = None
    print('In pca function')
    for samples, index in metric_logger.log_every(data_loader, 100):
        print(f'At the index {index[0]}')
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        if multiscale:
            feats = multi_scale(samples, model, args)
        else:

            if args.pt_style in ['clip', 'clip_wikiart']:
                allfeats = model.module.visual.get_intermediate_layers(samples.type(model.module.dtype))
                allfeats.reverse()
                if args.arch == 'resnet50':
                    raise Exception('code not written for this case')
                else:
                    temp = allfeats[args.layer - 1]
                    temp = torch.nn.functional.normalize(temp, dim=2)
                    # Doing gram matrix
                    feats = torch.einsum('bij,bik->bjk', temp, temp)
                    feats = feats.div(temp.shape[1])
                    feats = rearrange(feats, 'b c d -> b (c d)')
                    if pca_model is not None:
                        feats = feats.cpu().detach().numpy()
                        feats = pca_model.apply_py(feats)
                        feats = torch.from_numpy(feats).cuda().clone()
                    else:
                        feats = feats.detach().clone()
                    del temp
                del allfeats
            elif args.pt_style == 'vgg':
                temp = model.module.features(samples)
                temp = temp.view(temp.size(0), temp.size(1), -1)
                feats = torch.einsum('bji,bki->bjk', temp, temp)
                feats = feats.div(temp.shape[1])
                feats = rearrange(feats, 'b c d -> b (c d)')
                if pca_model is not None:
                    feats = feats.cpu().detach().numpy()
                    feats = pca_model.apply_py(feats)
                    feats = torch.from_numpy(feats).cuda().clone()
                else:
                    feats = feats.detach().clone()
                del temp
            else:
                raise Exception('Code not written for these ptstyles. Come back later.')

        feats = nn.functional.normalize(feats, dim=1, p=2).to(torch.float16)
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1], dtype=feats.dtype)
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    if pca_model is None:
        features = features.detach().numpy()
        pca = faiss.PCAMatrix(features.shape[-1], k)
        pca.train(features)
        trans_features = pca.apply_py(features)
        return torch.from_numpy(trans_features), pca
    else:
        return features, None


# saving features into numpy files
def save_embeddings_numpy(embeddings, filenames, savepath):
    os.makedirs(savepath, exist_ok=True)
    for c, fname in enumerate(filenames):
        np_emb = np.asarray(embeddings[c, :].cpu().detach(), dtype=np.float16)
        np.save(f'{savepath}/{fname}.npy', np_emb)
