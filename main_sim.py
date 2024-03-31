#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import os
import pathlib
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import torchvision.models as torchvision_models
from torchvision.models import VGG16_Weights

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import utils
from utils import extract_features_pca
from models import dino_vits, moco_vits
from data.wikiart import WikiArtD


parser = argparse.ArgumentParser('dynamicDistances-Embedding Generation Module')
parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset",
                    choices=['wikiart'])

parser.add_argument('--qsplit', default='query', choices=['query', 'database'], type=str, help="The inferences")
parser.add_argument('--data-dir', type=str, default=None,
                        help='The directory of concerned dataset')
parser.add_argument('--pt_style', default='csd', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--multiscale', default=False, type=utils.bool_flag)

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--num_loss_chunks', default=1, type=int)
parser.add_argument('--isvit', action='store_true')
parser.add_argument('--layer', default=1, type=int, help="layer from end to create descriptors from.")
parser.add_argument('--feattype', default='normal', type=str, choices=['otprojected', 'weighted', 'concated', 'gram', 'normal'])
parser.add_argument('--projdim', default=256, type=int)

parser.add_argument('-mp', '--model_path', type=str, default=None)
parser.add_argument('--gram_dims', default=1024, type=int)
parser.add_argument('--query_count', default=-1, type=int, help='Number of queries to consider for final evaluation. Works only for domainnet')

parser.add_argument('--embed_dir', default='./embeddings', type=str, help='Directory to save embeddings')

## Additional config for CSD
parser.add_argument('--eval_embed', default='head', choices=['head', 'backbone'], help="Which embed to use for eval")
parser.add_argument('--skip_val', action='store_true')


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # utils.init_distributed_mode(args)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    if args.pt_style == 'dino':
        dinomapping = {
            'vit_base': 'dino_vitb16',
            'vit_base8': 'dino_vitb8',  # TODO: this mapping is incorrect. Change it later
        }
        if args.arch not in dinomapping:
            raise NotImplementedError('This model type does not exist/supported for DINO')
        model = dino_vits.__dict__[dinomapping[args.arch]](
                pretrained=True
            )
    elif args.pt_style == 'moco':
        if args.arch == 'vit_base':
            model = moco_vits.__dict__[args.arch]()
            pretrained = torch.load('./pretrainedmodels/vit-b-300ep.pth.tar', map_location='cpu')
            state_dict = pretrained['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError('This model type does not exist/supported for MoCo')
    elif args.pt_style == 'clip':
        from models import clip
        clipmapping = {
            'vit_large': 'ViT-L/14',
            'vit_base': 'ViT-B/16',
        }
        if args.arch not in clipmapping:
            raise NotImplementedError('This model type does not exist/supported for CLIP')
        model, preprocess = clip.load(clipmapping[args.arch])
    elif args.pt_style == 'vgg':
        model = torchvision_models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif args.pt_style == 'sscd':
        if args.arch == 'resnet50':
            model = torch.jit.load("./pretrainedmodels/sscd_disc_mixup.torchscript.pt")
        elif args.arch == 'resnet50_disc':
            model = torch.jit.load("./pretrainedmodels/sscd_disc_large.torchscript.pt")
        else:
            NotImplementedError('This model type does not exist/supported for SSCD')
    elif args.pt_style.startswith('csd'):
        assert args.model_path is not None, "Model path missing for CSD model"
        from CSD.model import CSD_CLIP
        from CSD.utils import has_batchnorms, convert_state_dict
        from CSD.loss_utils import transforms_branch0

        args.content_proj_head = "default"
        model = CSD_CLIP(args.arch, args.content_proj_head)
        if has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        checkpoint = torch.load(args.model_path, map_location="cpu")
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"=> loaded checkpoint with msg {msg}")
        preprocess = transforms_branch0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    if args.pt_style == 'clip':  # and args.arch == 'resnet50':
        ret_transform = preprocess
    elif args.pt_style.startswith('csd'):
        ret_transform = preprocess
    elif args.pt_style in ['dino', 'moco', 'vgg']:
        ret_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        ret_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    if args.dataset == 'wikiart':
        dataset_query = WikiArtD(args.data_dir, args.qsplit, ret_transform)
        dataset_values = WikiArtD(args.data_dir, 'database', ret_transform)
    else:
        raise NotImplementedError

    ## creating dataloader
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_values, shuffle=False)
        qsampler = torch.utils.data.distributed.DistributedSampler(dataset_query, shuffle=False)
    else:
        sampler = None
        qsampler = None
    data_loader_values = torch.utils.data.DataLoader(
        dataset_values,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        sampler=qsampler,
        batch_size=args.batch_size if args.feattype != 'gram' else 32,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"train: {len(dataset_values)} imgs / query: {len(dataset_query)} imgs")
    model.eval()

    ############################################################################
    if not args.multiprocessing_distributed:
        utils.init_distributed_mode(args)
    if args.rank == 0:  # only rank 0 will work from now on

        # Step 1: extract features
        os.makedirs(args.embed_dir, exist_ok=True)
        embsavepath = os.path.join(
            args.embed_dir,
            f'{args.pt_style}_{args.arch}_{args.dataset}_{args.feattype}',
            f'{str(args.layer)}')
        if args.feattype == 'gram':
            path1, path2 = embsavepath.split('_gram')
            embsavepath = '_'.join([path1, 'gram', str(args.gram_dims), args.qsplit, path2])

        if os.path.isfile(os.path.join(embsavepath, 'database/embeddings_0.pkl')) or args.skip_val:
            valexist = True
        else:
            valexist = False
        if args.feattype == 'gram':
            pca_dirs, meanvals = None, None
            query_features, pca_dirs = extract_features_pca(args, model, pca_dirs, args.gram_dims, data_loader_query,
                                                                    False, multiscale=args.multiscale)
            if not valexist:
                values_features, _ = extract_features_pca(args, model, pca_dirs, args.gram_dims, data_loader_values,
                                                          False, multiscale=args.multiscale)

        elif args.pt_style.startswith('csd'):
            from CSD.utils import extract_features
            query_features = extract_features(model, data_loader_query, use_cuda=False, use_fp16=True, eval_embed=args.eval_embed)

            if not valexist:
                values_features = extract_features(model, data_loader_values, use_cuda=False, use_fp16=True, eval_embed=args.eval_embed)
        else:
            from utils import extract_features
            query_features = extract_features(args, model, data_loader_query, False, multiscale=args.multiscale)
            if not valexist:
                values_features = extract_features(args, model, data_loader_values, False,
                                                   multiscale=args.multiscale)

        from search.embeddings import save_chunk
        l_query_features = list(np.asarray(query_features.cpu().detach(), dtype=np.float16))

        save_chunk(l_query_features, dataset_query.namelist, 0, f'{embsavepath}/{args.qsplit}')
        if not valexist:
            l_values_features = list(np.asarray(values_features.cpu().detach(), dtype=np.float16))
            save_chunk(l_values_features, dataset_values.namelist, 0, f'{embsavepath}/database')

        print(f'Embeddings saved to: {embsavepath}')


if __name__ == '__main__':
    main()
