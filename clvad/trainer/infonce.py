import os
import time
import random
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from clvad.trainer.base import Trainer
import clvad.utils.tensorboard_utils as TB
import clvad.utils.transforms as T
from clvad.feature.utils.utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint, load_checkpoint)


class InfonceTrainer(Trainer):
    def __init__(self, args, model):
        (self.local_rank,
         self.gpu,
         self.data_config,
         self.log_config,
         self.opt_config) = self.__parse_args(args)
        self.epoch = self.opt_config['start_epoch']
        # set logger#
        self.writer_train = SummaryWriter(logdir=self.log_config['log_path'])
        self.train_plotter = TB.PlotterThread(self.writer_train)

        self.model = model
        self.optimizer = self.__create_optimizer(self.opt_config)
        self.criterion = self.__create_criterion(self.gpu)
        self.scheduler = self.__create_scheduler(
            self.optimizer, self.opt_config)
        self.epoch = 0
        self.iteration = 1
        self.min_loss = 1e10
        self.is_best = False
        self.transform_train_cuda = transforms.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225], channel=1)])

    def load_ckpt(self, args):
        load_checkpoint(args)
        # load epoch

    def save_ckpt(self, save_dict,
                  file_name=None, save_path=None, save_dir=None):
        save_dict = {
            'epoch': self.epoch,
            'state_dict': self.model.module.state_dict(),
            'min_loss': self.min_loss,
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration}
        file_name = file_name if file_name is not None \
            else 'epoch{self.epoch}.pth.tar'
        save_dir = save_dir if save_dir is not None \
            else self.log_config['log_path']
        save_path = save_path if save_path is not None \
            else os.path.join(save_dir, file_name)

        save_checkpoint(save_dict, self.is_best,
                        gap=self.log_config['save_freq'],
                        filename=save_path,
                        keep_all=True)

    def train(self, train_loader, logger):
        # main loop #
        while self.epoch < self.opt_config['end_epoch']:
            np.random.seed(self.epoch)
            random.seed(self.epoch)
            train_loader.sampler.set_epoch(self.epoch)
            self.scheduler.step()
            if self.epoch in self.opt_config['schedule']:
                logger.info('\tLR scheduler: new learning rate is %g'
                            % float(self.scheduler.get_lr()[0]))

            self.__train_one_epoch(
                train_loader, self.model,
                self.criterion, self.optimizer, self.epoch,
                logger)

            if (self.epoch % self.log_config['save_freq'] == 0) or \
                    (self.epoch == self.opt_config['end_epoch'] - 1):
                # save check_point on rank==0 worker
                if self.local_rank == 0:
                    self.save_ckpt()
            self.epoch += 1
        # end #

        logger.info('Training from ep %d to ep %d finished' %
                    (self.opt_config['start_epoch'],
                     self.opt_config['end_epoch']))

    def __train_one_epoch(self,
                          data_loader,
                          model,
                          criterion,
                          optimizer,
                          epoch,
                          logger):
        # about logger #
        batch_time = AverageMeter('Time', ':.2f')
        data_time = AverageMeter('Data Time', ':.2f')
        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses],
            prefix='Epoch:[{}]'.format(epoch))

        # Start Training! #
        model.train()
        tic = time.time()
        end = time.time()
        for idx, (input_seq, _) in tqdm(enumerate(data_loader)):
            data_time.update(time.time() - end)
            B = input_seq.size(0)
            input_seq = self.__tr(input_seq.cuda(non_blocking=True))

            output, target = model(input_seq)
            loss = criterion(output, target)
            losses.update(loss.item(), B)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(idx)
            if idx % self.log_config['print_freq'] == 0:
                if self.log_config['print']:
                    self.train_plotter.add_data(
                        'local/loss', losses.local_avg, self.iteration)

            self.iteration += 1

        if self.print:
            logger.info('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
                        'T-epoch:{t:.2f}\t'.format(
                            epoch, idx, len(data_loader),
                            gpu=self.local_rank, t=time.time()-tic))

        if self.print:
            self.train_plotter.add_data('global/loss', losses.avg, epoch)

        self.is_best = losses.avg < self.min_loss
        self.min_loss = min(self.min_loss, losses.avg)
        return losses.avg

    def __tr(self, x):
        B = x.size(0)
        return (self.transforms_train_cuda(x)
                .view(B, 3, self.data_config['num_seq'],
                      self.data_config['seq_len'],
                      self.data_config['img_dim'], self.data_config['img_dim'])
                .transpose(1, 2).contiguous())

    def __parse_args(self, args):
        local_rank = args.local_rank
        gpu = args.gpu
        data_config = {
            'num_seq': args.num_seq,
            'seq_len': args.seq_len,
            'img_dim': args.img_dim
        }
        log_config = {
            'print': args.print,
            'print_freq': args.print_freq,
            'save_freq': args.save_freq,
            'log_path': args.log_path
        }
        opt_config = {
            'opt_func': optim.Adam,
            'lr': args.lr,
            'wd': args.wd,
            'schedule': args.schedule,
            'start_epoch': args.start_epoch,
            'end_epoch': args.epochs
        }
        return local_rank, gpu, data_config, log_config, opt_config

    def __create_criterion(self, gpu):
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        return criterion

    def __create_optimizer(self, opt_config):
        # optimizer #
        params = []
        for _, param in self.model.named_parameters():
            params.append({'params': param})
        optimizer = opt_config['opt_func'](
            params,
            lr=opt_config['lr'],
            weight_decay=opt_config['wd'])
        return optimizer

    def __create_scheduler(self, optimizer, opt_config):
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=opt_config['schedule'], gamma=0.1)
        return scheduler
