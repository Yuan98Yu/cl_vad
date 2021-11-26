import glob
from PIL import Image

import numpy as np
import torch
# from torch import nn

# from clvad.classifier.data.base import BaseADDataset
from clvad.feature.data.dataset import SHTECH_2CLIP


class SHTECHDataset(SHTECH_2CLIP):
    def __init__(self, **kwargs):
        super(SHTECHDataset, self).__init__(**kwargs)
        # print(self.mode)

    def __getitem__(self, index):
        label = np.ones(self.num_frames)
        if self.mode == 'train':
            vpath, vlen = self.video_df.iloc[index]
        else:
            vpath, vlen, label_file = self.video_df.iloc[index]
            label = np.load(label_file)
        img_list = sorted(list(glob.glob(vpath + '/*.jpg')))
        assert len(img_list) == vlen

        frame_index = self.frame_sampler(
            vlen) if self.mode == 'train' else np.arange(vlen)
        seq = [Image.open(img_list[i]) for i in frame_index]

        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, 1)
        label = label[frame_index]
        # print(label.shape)

        return seq, label


# class CLFeatureDataset(BaseADDataset):
#     """A dataset which loads videos and returns features extracted by a model
#     """

#     def __init__(self, vad_dataset: SHTECH_2CLIP, cl_model: nn.Module):
#         super(CLFeatureDataset, self).__init__()
#         self.vad_dataset = vad_dataset
#         self.cl_model = cl_model

#     def __getitem__(self, index):
#         """Override the original method of the MNIST class.
#         Args:
#             index (int): Index
#         Returns:
#             triple: (image, target, index) where target is index of the target class.
#         """

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         seq, label = self.vad_dataset[index]

#         feature = self.cl_model(seq)
#         if self.target_transform is not None:
#             target = self.target_transform(label)

#         return seq, label, index  # only line changed

#     def __len__(self):
#         return len(self.vad_dataset)
