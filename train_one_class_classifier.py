import sys
import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
# import torch.nn as nn
# import torch.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
# import matplotlib.pyplot as plt

from clvad.classifier.model.factory import create_model
from clvad.classifier.data.factory import create_dataset, create_transform
from clvad.classifier.utils.optim import (adjust_learning_rate, get_radius,
                                          init_center_c)
from clvad.classifier.utils.meter import AverageMeter, ProgressMeter
from clvad.classifier.utils.checkpoint import save_checkpoint, load_checkpoint
import clvad.utils.distribute as distributed
from clvad.classifier.utils.options import parse_args
from clvad.utils.logger import set_logger
from clvad.classifier.utils.seed import set_seed
import clvad.utils.tensorboard_utils as TB
import clvad.utils.transforms as T
from clvad.utils.config import save_args, load_args


def train_one_epoch(data_loader, model, transforms_cuda, optimizer, epoch,
                    args, logger):
    def tr(x):  # transformation on tensor
        B = x.size(0)
        return transforms_cuda(x).view(B, 3, args.num_seq, args.seq_len,
                                       args.img_dim, args.img_dim) \
            .transpose(1, 2).contiguous()

    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data Time', ':.2f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(data_loader), [batch_time, data_time, losses],
                             prefix='Epoch:[{}]'.format(epoch))

    model.train()

    tic = time.time()
    end = time.time()

    for idx, (x, label) in tqdm(enumerate(data_loader),
                                total=len(data_loader),
                                disable=True):
        data_time.update(time.time() - end)
        B = x.size(0)
        x = tr(x.cuda(non_blocking=True))
        x = x.squeeze(1)

        y_hat, _ = model(x)
        dist = torch.sum((y_hat - args.c)**2, dim=1)
        if args.objective == 'soft-boundary':
            scores = dist - args.R**2
            loss = args.R ** 2 + \
                (1 / args.nu) * \
                torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update hypersphere radius R on mini-batch distances
        if (args.objective == 'soft-boundary') \
                and (epoch >= args.warm_up_n_epochs):
            args.R.data = torch.tensor(get_radius(dist, args.nu)).cuda()
        losses.update(loss.item(), B)

        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(idx)
        if idx % args.print_freq == 0:
            if args.print:
                args.train_plotter.add_data('local/loss', losses.local_avg,
                                            args.iteration)

        args.iteration += 1

    logger.info('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
                'T-epoch:{t:.2f}\t'.format(epoch,
                                           idx,
                                           len(data_loader),
                                           gpu=args.local_rank,
                                           t=time.time() - tic))

    if args.print:
        args.train_plotter.add_data('global/loss', losses.avg, epoch)

    return losses.avg


def main_worker(gpu, ngpus_per_node, args, logger):
    args.gpu = gpu

    distributed.init_distributed_env(args, ngpus_per_node, gpu)

    # model #
    model, params = create_model(args)
    model = distributed.transform_model_to_distributed_model(args, model)
    model_without_ddp = model.module

    # data #
    transform = create_transform(args)
    dataset = create_dataset(args, transform)
    transform_train_cuda = transforms.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    channel=1)
    ])
    # under coding
    # train_loader = create_dataloader(dataset, args)
    train_loader = distributed.transform_dataset_to_distributed_dataloader(
        args, dataset)

    # optimizer #
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    #    amsgrad=args.optimizer_name == 'amsgrad')
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # Set learning rate scheduler
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    # milestones=self.lr_milestones, gamma=0.1)

    # restart training #
    args.iteration = 1
    min_loss = 1e10
    args.c = None
    args.R = torch.tensor(args.R).cuda()
    if args.pretrain:
        load_checkpoint(args, model_without_ddp, optimizer)

    # Initialize hypersphere center c (if c not loaded)
    if args.c is None:
        logger.info('Initializing center c...')
        args.c = init_center_c(train_loader, model)
        logger.info('Center c initialized.')
    # Initialize DeepSVDD model and set neural network \phi
    assert args.objective in (
        'one-class', 'soft-boundary'
    ), "Objective must be either 'one-class' or 'soft-boundary'."
    assert (0 < args.nu) & (
        args.nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=args.log_path)
    args.train_plotter = TB.PlotterThread(writer_train)

    # main loop #
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train_one_epoch(train_loader, model, transform_train_cuda,
                                     optimizer, epoch, args, logger)

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            # save check_point on rank==0 worker
            if args.local_rank == 0:
                is_best = train_loss < min_loss
                min_loss = min(train_loss, min_loss)
                state_dict = model_without_ddp.state_dict()
                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'c': args.c,
                    'nu': args.nu,
                    'R': args.R,
                    'min_loss': min_loss,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration
                }
                save_checkpoint(save_dict,
                                is_best,
                                gap=args.save_freq,
                                filename=os.path.join(
                                    args.log_path, 'epoch%d.pth.tar' % epoch),
                                keep_all=True)

    logger.info('Training from ep %d to ep %d finished' %
                (args.start_epoch, args.epochs))
    sys.exit(0)


def main():
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    args = parse_args()
    if args.load_config:
        load_args(args.cfg_path, args)
    else:
        save_args(args.cfg_path, args)
    logger = set_logger(args.log_path)
    # Print configuration
    logger.info('Deep SVDD objective: %s' % args.objective)
    logger.info('Nu-paramerter: %.2f' % args.nu)
    args.logger = logger
    # # Default device to 'cpu' if cuda is not available
    # if not torch.cuda.is_available():
    #     device = 'cpu'

    # Set seed
    if args.seed != -1:
        logger.info('Set seed to %d.' % args.seed)
        set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args, logger)


if __name__ == '__main__':
    main()
