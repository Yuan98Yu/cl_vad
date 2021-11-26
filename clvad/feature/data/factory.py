import torchvision.transforms as transforms

from clvad.feature.data.dataset import SHTECH_2CLIP
import clvad.utils.augmentation as A
import clvad.utils.transforms as T


def create_transform(args, mode='train'):
    # seq_len = args.seq_len * 2  # for both rgb and flow
    seq_len = args.seq_len  # for both rgb and flow

    null_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False,
                          seq_len=seq_len, bottom_area=0.2),
        A.RandomHorizontalFlip(consistent=False, seq_len=seq_len),
        A.ToTensor(),
    ])

    base_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False,
                          seq_len=seq_len, bottom_area=0.2),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0,
                          consistent=False, seq_len=seq_len)
        ], p=0.8),
        A.RandomGray(p=0.2, seq_len=seq_len),
        transforms.RandomApply(
            [A.GaussianBlur([.1, 2.], seq_len=seq_len)], p=0.5),
        A.RandomHorizontalFlip(consistent=False, seq_len=seq_len),
        A.ToTensor(),
    ])

    # oneclip: temporally take one clip, random augment twice
    # twoclip: temporally take two clips, random augment for each
    # merge oneclip & twoclip transforms with 50%/50% probability
    transform = A.TransformController(
        [A.TwoClipTransform(base_transform, null_transform, seq_len=seq_len, p=0.3),
         A.OneClipTransform(base_transform, null_transform, seq_len=seq_len)],
        weights=[0.5, 0.5])
    print(transform)

    return transform


def create_dataset(args, transform, mode='train'):
    print('Loading data for "%s" mode' % mode)

    if args.dataset == 'shtech':
        dataset = SHTECH_2CLIP(split_file='./data/train_split01.csv', mode=mode, transform=transform,
                               num_frames=args.seq_len, ds=args.ds,
                               )
    else:
        raise NotImplementedError

    return dataset
