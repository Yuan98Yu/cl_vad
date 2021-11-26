import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # path #
    parser.add_argument('--log_path',
                        default='log/train_cl',
                        type=str,
                        help='path of model to resume')
    parser.add_argument('--cfg_path',
                        default='./log/train_cl/config.json',
                        type=str)
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path of model to resume')

    parser.add_argument('--dataset', default='shtech', type=str)
    

    # test #
    parser.add_argument(
        '--test10crop',
        action='store_true')
    # env #
    parser.add_argument('--gpu', default=None, type=str)
    args = parser.parse_args()
    return args
