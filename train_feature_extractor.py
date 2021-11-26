import clvad.utils.distribute as distributed
import os
import sys
import numpy as np
import random

import torch
import matplotlib.pyplot as plt

from clvad.feature.options import parse_args
from clvad.feature.model.factory import create_model
from clvad.feature.data.factory import create_dataset, create_transform
from clvad.trainer.factory import create_trainer
from clvad.utils.logger import set_logger

plt.switch_backend('agg')


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    distributed.init_distributed_env(args, ngpus_per_node, gpu)

    logger = set_logger(args.log_path)

    # model #
    model = create_model(args)
    model = distributed.transform_model_to_distributed_model(args, model)

    # data #
    train_transform = create_transform(args)
    train_dataset = create_dataset(args, train_transform)
    train_loader = distributed.transform_dataset_to_distributed_dataloader(
        args, train_dataset)

    # trainer #
    trainer = create_trainer(args)
    trainer.train(args, model, train_loader, logger)

    sys.exit(0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
