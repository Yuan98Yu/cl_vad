import os
import sys
import time
import json

import torch
import torch.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import clvad.utils.augmentation as A
import clvad.utils.transforms as T
from clvad.classifier.data.factory import create_dataset, create_transform
from clvad.classifier.model.factory import create_model
from clvad.classifier.utils.checkpoint import load_checkpoint
from clvad.classifier.utils.logger import set_logger
from clvad.classifier.utils.options import parse_args
from clvad.classifier.utils.seed import set_seed
from clvad.utils.config import load_args


def get_tr(args, transforms_cuda):
    def tr(x):
        B = x.size(0)
        assert B == 1
        num_test_sample = x.size(2)//(args.seq_len*args.num_seq)
        return transforms_cuda(x)\
            .view(3, num_test_sample, args.num_seq, args.seq_len,
                  args.img_dim, args.img_dim).permute(1, 2, 0, 3, 4, 5)
    return tr


# def summarize_probability(prob_dict, action_to_idx, title):
#     acc = [AverageMeter(), AverageMeter()]
#     stat = {}
#     for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
#         try:
#             action_name = vname.split('/')[-3]
#         except Exception:
#             action_name = vname.split('/')[-2]
#         target = action_to_idx(action_name)
#         mean_prob = torch.stack(item['mean_prob'], 0).mean(0)
#         mean_top1, mean_top5 = calc_topk_accuracy(mean_prob, torch.LongTensor([target]).cuda(), (1,5))
#         stat[vname] = {'mean_prob': mean_prob.tolist()}
#         acc[0].update(mean_top1.item(), 1)
#         acc[1].update(mean_top5.item(), 1)

#     print('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
#           .format(acc=acc))

#     with open(os.path.join(os.path.dirname(args.test),
#         '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
#         json.dump(stat, fp)
#     return acc


def test(args, test_loader, transforms_cuda, model, logger):
    '''Under coding...
    '''
    tr = get_tr(args, transforms_cuda)

    logger.info('Starting testing...')
    start_time = time.time()
    idx_label_score = []
    score_list = list()
    model.eval()
    with torch.no_grad():
        for input_seq, labels in test_loader:
            input_seq = tr(input_seq.to(args.device, non_blocking=True))
            input_seq = input_seq.squeeze(1)  # num_seq is always 1, seqeeze it
            logit, _ = model(input_seq)
            dist = torch.sum((logit - args.c) ** 2, dim=1)
            if args.objective == 'soft-boundary':
                score = dist - args.R ** 2
            else:
                score = dist
            score_list.append(score)
            # # Save triples of (idx, label, score) in a list
            # idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
            #                             labels.cpu().data.numpy().tolist(),
            #                             scores.cpu().data.numpy().tolist()))

    args.test_time = time.time() - start_time
    logger.info('Testing time: %.3f' % args.test_time)

    args.test_scores = idx_label_score

    # Compute AUC
    # _, labels, scores = zip(*idx_label_score)
    # labels = np.array(labels)
    # scores = np.array(scores)

    # self.test_auc = roc_auc_score(labels, scores)
    # logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

    logger.info('Finished testing.')


# def test_10crop(dataset, model, criterion,
#                 transforms_cuda, device, epoch, args, logger):
#     tr = get_tr(args, transforms_cuda)
#     prob_dict = {}
#     model.eval()

#     # aug_list: 1,2,3,4,5 = topleft, topright, bottomleft, bottomright, center
#     # flip_list: 0,1 = raw, flip
#     if args.center_crop:
#         print('Test using center crop')
#         args.logger.log('Test using center_crop\n')
#         aug_list = [5]
#         flip_list = [0]
#         title = 'center'
#     if args.five_crop:
#         print('Test using 5 crop')
#         args.logger.log('Test using 5_crop\n')
#         aug_list = [5, 1, 2, 3, 4]
#         flip_list = [0]
#         title = 'five'
#     if args.ten_crop:
#         print('Test using 10 crop')
#         args.logger.log('Test using 10_crop\n')
#         aug_list = [5, 1, 2, 3, 4]
#         flip_list = [0, 1]
#         title = 'ten'

#     def tr(x):
#         B = x.size(0)
#         assert B == 1
#         num_test_sample = x.size(2)//(args.seq_len*args.num_seq)
#         return transforms_cuda(x) \
#             .view(3, num_test_sample, args.num_seq, args.seq_len, args.img_dim, args.img_dim).permute(1, 2, 0, 3, 4, 5)

#     with torch.no_grad():
#         end = time.time()
#         # for loop through 10 types of augmentations,
#         # then average the probability
#         for flip_idx in flip_list:
#             for aug_idx in aug_list:
#                 print('Aug type: %d; flip: %d' % (aug_idx, flip_idx))
#                 if flip_idx == 0:
#                     transform = transforms.Compose([
#                         A.RandomHorizontalFlip(command='left'),
#                         A.FiveCrop(size=(224, 224), where=aug_idx),
#                         A.Scale(size=(args.img_dim, args.img_dim)),
#                         A.ColorJitter(0.2, 0.2, 0.2, 0.1,
#                                       p=0.3, consistent=True),
#                         A.ToTensor()])
#                 else:
#                     transform = transforms.Compose([
#                         A.RandomHorizontalFlip(command='right'),
#                         A.FiveCrop(size=(224, 224), where=aug_idx),
#                         A.Scale(size=(args.img_dim, args.img_dim)),
#                         A.ColorJitter(0.2, 0.2, 0.2, 0.1,
#                                       p=0.3, consistent=True),
#                         A.ToTensor()])

#                 dataset.transform = transform
#                 dataset.return_path = True
#                 dataset.return_label = True
#                 test_sampler = data.SequentialSampler(dataset)
#                 data_loader = data.DataLoader(dataset,
#                                               batch_size=1,
#                                               sampler=test_sampler,
#                                               shuffle=False,
#                                               num_workers=args.workers,
#                                               pin_memory=True)

#                 for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
#                     input_seq = tr(input_seq.to(device, non_blocking=True))
#                     # num_seq is always 1, seqeeze it
#                     input_seq = input_seq.squeeze(1)
#                     logit, _ = model(input_seq)

#                     # average probability along the temporal window
#                     prob_mean = F.softmax(logit, dim=-1).mean(0, keepdim=True)

#                     target, vname = target
#                     vname = vname[0]
#                     if vname not in prob_dict.keys():
#                         prob_dict[vname] = {'mean_prob': [], }
#                     prob_dict[vname]['mean_prob'].append(prob_mean)

#                 if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
#                     print('center-crop result:')
#                     acc_1 = summarize_probability(prob_dict,
#                                                   data_loader.dataset.encode_action, 'center')
#                     logger.info('center-crop:')
#                     logger.info('test Epoch: [{0}]\t'
#                                 'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
#                                 .format(epoch, acc=acc_1))

#             if (title == 'ten') and (flip_idx == 0):
#                 print('five-crop result:')
#                 acc_5 = summarize_probability(
#                     prob_dict,
#                     data_loader.dataset.encode_action,
#                     'five'
#                     )
#                 logger.info('five-crop:')
#                 logger.info('test Epoch: [{0}]\t'
#                             'Mean: Acc@1: {acc[0].avg:.4f} '
#                             'Acc@5: {acc[1].avg:.4f}'
#                             .format(epoch, acc=acc_5))

#     print('%s-crop result:' % title)
#     acc_final = summarize_probability(prob_dict,
#                                       data_loader.dataset.encode_action, 'ten')
#     logger.info('%s-crop:' % title)
#     logger.info('test Epoch: [{0}]\t'
#                 'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
#                 .format(epoch, acc=acc_final))
#     sys.exit(0)


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

    # data #
    transform = create_transform(args)
    dataset = create_dataset(args, transform)
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
