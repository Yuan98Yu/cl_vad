from torchvision import transforms

import clvad.utils.augmentation as A
from clvad.classifier.data.base import BaseADDataset
from clvad.classifier.data.cl_shtech import SHTECHDataset


def create_transform(args, mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            A.RandomSizedCrop(size=224, consistent=True, bottom_area=0.2),
            A.Scale(args.img_dim),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True),
            A.ToTensor(),
        ])
    elif mode == 'val' or mode == 'test':
        transform = transforms.Compose([
            A.RandomSizedCrop(size=224, consistent=True, bottom_area=0.2),
            A.Scale(args.img_dim),
            A.ToTensor(),
        ])
    return transform


def create_dataset(args, transform, mode='train') -> BaseADDataset:
    if args.dataset == 'shtech':
        dataset = SHTECHDataset(
            split_file='./data/train_split01.csv',
            mode=mode,
            transform=transform,
            num_frames=args.seq_len,
            ds=args.ds,
        )
    else:
        raise NotImplementedError
    return dataset


# def create_dataloader(dataset, batch_size: int, shuffle, num_workers: int = 0) -> DataLoader:
#     data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
#                              num_workers=num_workers)

#     return data_loader
