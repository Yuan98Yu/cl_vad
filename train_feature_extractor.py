import clvad.utils.distribute as distributed
import os
import sys
import time
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from clvad.feature_learning.options import parse_args
import clvad.utils.tensorboard_utils as TB
import clvad.utils.transforms as T
from clvad.feature_learning.utils.utils import (
    adjust_learning_rate,
    AverageMeter,
    ProgressMeter,
    save_checkpoint, load_checkpoint)
from clvad.feature_learning.model.factory import create_model
from clvad.feature_learning.data.factory import create_dataset, create_transform

plt.switch_backend('agg')


def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, epoch, args):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data Time', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))

    model.train()

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim)\
            .transpose(1, 2).contiguous()

    tic = time.time()
    end = time.time()

    for idx, (input_seq, label) in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
        data_time.update(time.time() - end)
        B = input_seq.size(0)
        input_seq = tr(input_seq.cuda(non_blocking=True))

        if args.model == 'infonce':  # 'target' is the index of self
            output, target = model(input_seq)
            loss = criterion(output, target)

        if args.model == 'ubernce':  # 'target' is the binary mask
            label = label.cuda(non_blocking=True)
            output, target = model(input_seq, label)
            # optimize all positive pairs, compute the mean for num_pos and for batch_size
            loss = - (F.log_softmax(output, dim=1) *
                      target).sum(1) / target.sum(1)
            loss = loss.mean()

        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(idx)
        if idx % args.print_freq == 0:
            if args.print:
                args.train_plotter.add_data(
                    'local/loss', losses.local_avg, args.iteration)

        args.iteration += 1

    print('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), gpu=args.local_rank, t=time.time()-tic))

    if args.print:
        args.train_plotter.add_data('global/loss', losses.avg, epoch)

    return losses.avg


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    distributed.init_distributed_env(args, ngpus_per_node, gpu)

    # model #
    model = create_model(args)
    model = distributed.transform_model_to_distributed_model(args, model)
    model_without_ddp = model.module

    # data #
    train_transform = create_transform(args)
    transform_train_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], channel=1)])
    train_dataset = create_dataset(args, train_transform)
    train_loader = distributed.transform_dataset_to_distributed_dataloader(
        args, train_dataset)

    # optimizer #
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})
    # print('\n===========Check Grad============')
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name, param.requires_grad)
    # print('=================================\n')
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    # restart training #
    min_loss = 1e10
    load_checkpoint(args, model_without_ddp, optimizer)

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=args.log_path)
    args.train_plotter = TB.PlotterThread(writer_train)

    # main loop #
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, transform_train_cuda, epoch, args)

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            # save check_point on rank==0 worker
            if args.local_rank == 0:
                is_best = train_loss < min_loss
                min_loss = min(train_loss, min_loss)
                state_dict = model_without_ddp.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'min_loss': min_loss,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}
                save_checkpoint(save_dict, is_best, gap=args.save_freq,
                                filename=os.path.join(
                                    args.log_path, 'epoch%d.pth.tar' % epoch),
                                keep_all=True)

    print('Training from ep %d to ep %d finished' %
          (args.start_epoch, args.epochs))
    sys.exit(0)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    '''
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank

    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    '''
    args = parse_args()
    main(args)
