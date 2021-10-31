import glob
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import json

import torch
import pandas as pd
import numpy as np


# naming convension:
# {}_2CLIP is for pretraining
# without 2CLIP is for action classification

__all__ = [
    'SHTECH_2CLIP'
]


def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content


def pil_from_raw_rgb(raw):
    return Image.open(BytesIO(raw)).convert('RGB')


def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


class SHTECH_2CLIP(object):
    def __init__(self,
                 split_file,
                 transform=None,
                 mode='val',
                 num_frames=32,
                 ds=1,
                 window=False):
        self.split_file = split_file
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.window = window
        self.ds = ds

        split_mode = mode
        if mode == 'val':
            split_mode = 'test'
        video_df = pd.read_csv(split_file)

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_df.iterrows(), total=len(video_df), disable=True):
            vpath, vlen = row
            if vlen-self.num_frames//2*self.ds-1 <= 0:  # allow max padding = half video
                drop_idx.append(idx)
        print(f'drop too short videos: {drop_idx}')
        self.video_df = video_df.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_df = self.video_df.sample(
                frac=0.3, random_state=666)

    def frame_sampler(self, total):
        if (self.mode == 'test') or self.window:  # half overlap - 1
            if total-self.num_frames*self.ds <= 0:  # pad left, only sample once
                sequence = np.arange(self.num_frames)*self.ds
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                available = total-self.num_frames*self.ds
                start = np.expand_dims(
                    np.arange(0, available+1, self.num_frames*self.ds//2-1), 1)
                # [test_sample, num_frames]
                seq_idx = np.expand_dims(
                    np.arange(self.num_frames)*self.ds, 0) + start
                seq_idx = seq_idx.flatten(0)
        else:  # train or val
            if total-self.num_frames*self.ds <= 0:  # pad left
                sequence = np.arange(self.num_frames)*self.ds + \
                    np.random.choice(range(self.ds), 1)
                seq_idx = np.zeros_like(sequence)
                sequence = sequence[sequence < total]
                seq_idx[-len(sequence)::] = sequence
            else:
                start = np.random.choice(
                    range(total-self.num_frames*self.ds), 1)
                seq_idx = np.arange(self.num_frames)*self.ds + start
        return seq_idx

    def double_sampler(self, total):
        seq1 = self.frame_sampler(total)
        seq2 = self.frame_sampler(total)
        return np.concatenate([seq1, seq2])

    def __getitem__(self, index):
        label = 1
        vpath, vlen = self.video_df.iloc[index]
        img_list = sorted(list(glob.glob(vpath + '/*.jpg')))
        assert len(img_list) == vlen

        frame_index = self.double_sampler(vlen)
        seq = [Image.open(img_list[i]) for i in frame_index]

        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        return seq, label

    def __len__(self):
        return len(self.video_df)
