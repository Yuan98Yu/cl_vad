import numpy as np
import torch
from torch import optim


def create_optimizer(model, args):
    # optimizer #
    if args.train_what == 'last':
        print('=> [optimizer] only train last layer')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                params.append({'params': param})
    elif args.train_what == 'ft':
        print('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    else:  # train all
        params = []
        print('=> [optimizer] train all layer')
        for name, param in model.named_parameters():
            params.append({'params': param})
    if args.train_what == 'last':
        print('\n===========Check Grad============')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.requires_grad)
        print('=================================\n')

    if args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.wd,
                              momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer


def init_center_c(train_loader: torch.utils.data.DataLoader,
                  net: torch.nn.Module,
                  eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.module.output_dim).cuda()

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.cuda()
            outputs, _ = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
