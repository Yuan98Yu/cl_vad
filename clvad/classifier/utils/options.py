import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add('--strategy',
               default='infonce',
               type=str)
    # path #
    parser.add_argument('--log_path',
                        default='log/train_cl',
                        type=str,
                        help='path of model to resume')
    parser.add_argument('--load_config',
                        action='store_true',
                        help='Load a config file instead of parsing args?')
    parser.add_argument('--cfg_path',
                        default='./log/train_cl/config.json',
                        type=str)
    # model #
    parser.add_argument('--objective', default='soft-boundary', type=str)
    parser.add_argument('--nu', default=0.5, type=float, help='nu')
    parser.add_argument('--R', default=1e-3, type=float, help='R')
    parser.add_argument('--output_dim',
                        default=128,
                        type=int,
                        help='output dim')
    parser.add_argument('--model', default='lincls', type=str)
    parser.add_argument('--network', default='s3d', type=str)  # r18-all
    parser.add_argument('--train_what', default='last', type=str)
    parser.add_argument('--pretrain',
                        default='./log/train/epoch98000.pth.tar',
                        type=str)
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path of model to resume')

    # dataset #
    parser.add_argument('--dataset', default='shtech', type=str)
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--seq_len',
                        default=32,
                        type=int,
                        help='number of frames in each video block')
    parser.add_argument('--num_seq',
                        default=1,
                        type=int,
                        help='number of video blocks')
    parser.add_argument('--ds',
                        default=1,
                        type=int,
                        help='frame down sampling rate')

    # training #
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule',
                        default=[120, 160],
                        nargs='*',
                        type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float,
                        help='learning rate')
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='number of total epochs to run')
    parser.add_argument('--warm_up_n_epochs',
                        default=10,
                        type=int,
                        help='number of warm up epochs')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        help='manual epoch number (useful on restarts)')

    # test #
    parser.add_argument(
        '--test10crop',
        action='store_true')
    # env #
    parser.add_argument('--gpu', default=None, type=str)

    # log and save #
    parser.add_argument('--print_freq',
                        default=5,
                        type=int,
                        help='frequency of printing output during training')
    parser.add_argument('--save_freq',
                        default=1,
                        type=int,
                        help='frequency of eval')

    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # parallel configs #
    parser.add_argument('--dist-url',
                        default='env://',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')
    parser.add_argument(
        '--multiprocessing-distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training')

    # for torch.distributed.launch #
    parser.add_argument('--local_rank',
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()
    return args
