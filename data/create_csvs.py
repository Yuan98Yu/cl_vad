from glob import glob
from tqdm import tqdm

import pandas as pd


def main(root, save_path):
    df = list()
    for video_path in tqdm(glob(root + '/*')):
        # print(video_path)
        df.append((video_path, len(glob(video_path + '/*.jpg'))))
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(save_path)
    pass


if __name__ == '__main__':
    root = '/home/yuanyu/projects/datasets/ShanghaiTech/training/frames'
    save_path = '/home/yuanyu/projects/exps/CoCLR/process_data/data/shtech/train_split01.csv'
    main(root, save_path)
