import os
import glob

import torch


def load_checkpoint(args, model_without_ddp, optimizer=None):
    # restart training #
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            args.iteration = checkpoint['iteration']
            args.c = checkpoint['iteration']
            args.nu = checkpoint['nu']
            args.R = checkpoint['R']
            state_dict = checkpoint['state_dict']

            try:
                model_without_ddp.load_state_dict(state_dict)
            except Exception:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_ddp,
                                    state_dict,
                                    verbose=True)
            print("=> load resumed checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))

            if optimizer:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception:
                    print(
                        '[WARNING] failed to load optimizer state, '
                        'initialize optimizer'
                    )
        else:
            print("[Warning] no checkpoint found at '{}', use random init".
                  format(args.resume))

    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            state_dict = checkpoint['state_dict']

            new_dict = {}
            for k, v in state_dict.items():
                k = k.replace('encoder_q.0.', 'backbone.')
                new_dict[k] = v
            state_dict = new_dict

            try:
                model_without_ddp.load_state_dict(state_dict)
            except Exception:
                neq_load_customized(model_without_ddp,
                                    state_dict,
                                    verbose=True)
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(
                args.pretrain, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}', use random init".
                  format(args.pretrain))
            raise NotImplementedError

    else:
        print("=> train from scratch")


def save_checkpoint(state,
                    is_best=0,
                    gap=1,
                    filename='models/checkpoint.pth.tar',
                    keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(
        os.path.dirname(filename),
        'epoch%s.pth.tar' % str(state['epoch'] - gap))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except Exception:
            pass

    if is_best:
        past_best = glob.glob(
            os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best,
                           key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 5:
            try:
                os.remove(past_best[0])
            except Exception:
                pass
        torch.save(
            state,
            os.path.join(os.path.dirname(filename),
                         'model_best_epoch%s.pth.tar' % str(state['epoch'])))


def neq_load_customized(model, pretrained_dict, verbose=True):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
        for k, v in pretrained_dict.items():
            if k in model_dict:
                tmp[k] = v
            else:
                print(k)
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    # if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model
