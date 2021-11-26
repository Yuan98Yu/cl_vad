from glob import glob
from tqdm import tqdm

import pandas as pd


def main(root, save_path, label_root, mode='train'):
    df = list()
    for video_path in tqdm(sorted(glob(root + '/*'))):
        # print(video_path)
        if mode == 'train':
            df.append((video_path, len(glob(video_path + '/*.jpg'))))
        else:
            df.append((video_path,
                       len(glob(video_path + '/*.jpg')),
                       label_root + video_path.split('/')[-1]+'.npy')
                      )
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(save_path)
    pass


if __name__ == '__main__':
    root = '/home/yuanyu/projects/datasets/ShanghaiTech/testing/frames'
    save_path = '/home/yuanyu/projects/cl_vad/data/test_split01.csv'
    label_root = '/home/yuanyu/projects/datasets/ShanghaiTech/testing/test_frame_mask/'
    main(root, save_path, label_root, 'test')
