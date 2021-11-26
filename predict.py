import os
import time

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# import clvad.utils.augmentation as A
import clvad.utils.transforms as T
from clvad.classifier.data.factory import create_dataset, create_transform
from clvad.classifier.model.factory import create_model
from clvad.classifier.utils.checkpoint import load_checkpoint
from clvad.classifier.utils.logger import set_logger
from clvad.utils.test_options import parse_args
from clvad.classifier.utils.seed import set_seed
from clvad.utils.config import load_args


def get_tr(args, transforms_cuda):
    def tr(x):
        B = x.size(0)
        assert B == 1
        # print('xshape', x.shape)
        num_frames = x.size(2)
        # num_test_sample = x.size(2)//(args.seq_len*args.num_seq)
        return transforms_cuda(x)\
            .view(3, 1, args.num_seq, num_frames,
                  args.img_dim, args.img_dim).permute(1, 2, 0, 3, 4, 5)
    return tr


def test(args, test_loader, transforms_cuda, model, logger):
    '''Under coding...
    '''
    tr = get_tr(args, transforms_cuda)

    logger.info('Starting testing...')
    start_time = time.time()
    score_list = list()
    label_list = list()
    model.eval()
    with torch.no_grad():
        for input_seq, labels in tqdm(test_loader):
            input_seq = tr(input_seq.to(args.device, non_blocking=True))
            input_seq = input_seq.squeeze(1)  # num_seq is always 1, seqeeze it
            b, c, t, h, w = input_seq.shape
            # print(b, c, t, h, w)
            # print(labels.shape)
            # return
            for start in range(0, t-args.seq_len, args.seq_len):
                input = input_seq[:, :, start:start+args.seq_len, :, :]
                # print(input.shape)
                label = labels[:, start:start+args.seq_len]
                logit, _ = model(input)
                dist = torch.sum((logit - args.c) ** 2, dim=1)
                if args.objective == 'soft-boundary':
                    score = dist - args.R ** 2
                else:
                    score = dist
                score_list.append(score.item())
                label_list.append(label.squeeze(0).cpu().numpy())

    args.test_time = time.time() - start_time
    logger.info('Testing time: %.3f' % args.test_time)

    pred = np.repeat(np.array(score_list), args.seq_len)
    gt = np.array(label_list).reshape(-1)
    print(pred[:200], gt[:200])
    fpr, tpr, threshold = roc_curve(list(gt), pred)
    np.save(f'{args.log_path}/fpr.npy', fpr)
    np.save(f'{args.log_path}/tpr.npy', tpr)
    roc_auc = auc(fpr, tpr)
    print('auc : ' + str(roc_auc))
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    precision, recall, th = precision_recall_curve(list(gt), pred)
    pr_auc = auc(recall, precision)
    print('pr_auc : ' + str(pr_auc))
    np.save(f'{args.log_path}/precision.npy', precision)
    np.save(f'{args.log_path}/recall.npy', recall)
    logger.info('Finished testing.')


def main():
    args = parse_args()
    load_args(args.cfg_path, args)
    logger = set_logger(args)
    args.logger = logger
    if args.seed != -1:
        set_seed(args.seed)
    if args.gpu is None:
        args.device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = torch.device('cuda')

    # model #
    model, _ = create_model(args)
    load_checkpoint(args, model, None)
    print('c', args.c)
    print('nu', args.nu)
    print('R', args.R)

    # return

    # data #
    transform = create_transform(args)
    dataset = create_dataset(args, transform, 'test')
    transforms_cuda = transforms.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    channel=1)
    ])
    dataset.transform = transform
    dataset.return_path = True
    dataset.return_label = True
    test_sampler = data.SequentialSampler(dataset)
    data_loader = data.DataLoader(dataset,
                                  batch_size=1,
                                  sampler=test_sampler,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=True)
    if args.test10crop:
        pass
        # test_10crop()
    else:
        test(args, data_loader, transforms_cuda, model, logger)


if __name__ == '__main__':
    main()
