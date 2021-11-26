import builtins

import torch
import torch.utils.data as data
import torch.distributed as dist

from clvad.feature.data.dataloader import FastDataLoader


def init_distributed_env(args, ngpus_per_node, gpu):
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url)
    # local rank is assigned autoly by torch.distributed.launch
    print(f'local rank_{args.local_rank} start')
    args.print = args.gpu == 0
    # suppress printing if not master
    if args.local_rank != 0:
        print('suppress printing')

        def print_pass(*args):
            pass
        builtins.print = print_pass


def transform_dataset_to_distributed_dataloader(args, dataset, mode='train'):
    print('Creating data loaders for "%s" mode' % mode)
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    if mode == 'train':
        data_loader = FastDataLoader(
            dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler, drop_last=True)
    else:
        raise NotImplementedError
    print('"%s" dataset has size: %d' % (mode, len(dataset)))

    return data_loader


def transform_model_to_distributed_model(args, model):
    if args.local_rank is not None:
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    return model


def run_with_distributed(args, main_worker):
    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)
