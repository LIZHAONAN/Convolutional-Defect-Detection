import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from augmentation import *
from utils import *
import math


class CSVDataset(Dataset):
    def __init__(self, path_to_csv, window_size=64, non_pos_ratio=7, pos_per_image=8, num_class=2,
                 method='mixed', transform=None, multiscale=True, shuffle=True):
        assert (method in ['hard', 'uniform', 'mixed'])
        self.df = pd.read_csv(path_to_csv, sep=',')
        self.img_paths = self.df['path'].unique()

        self.window_size = window_size
        self.non_pos_ratio = non_pos_ratio
        self.pos_per_image = pos_per_image
        self.num_class = num_class
        self.method = method
        self.multiscale = multiscale
        self.transform = transform
        self.shuffle = shuffle

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # load image
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('L')
        w, h = img.size
        # load labels
        lbl = self.df[self.df['path'] == img_path]
        lbl = lbl[['x', 'y', 'class']].values

        # sample true positive from labels
        pos_mask = np.random.choice(lbl.shape[0], self.pos_per_image)
        pos = np.zeros((self.pos_per_image, 3))  # [x, y, classes]

        pos[:, :2] = lbl[pos_mask, :2]  # x and y
        pos[:, 2] = lbl[pos_mask, 2]
        #         pos[np.arange(pos.shape[0]), 2 + lbl[pos_mask, 2].astype(int)] = 1
        # sample non positive
        non_pos = np.zeros((self.non_pos_ratio * self.pos_per_image, 3))
        for idx, _ in enumerate(non_pos):
            selected = None if self.method == 'uniform' else pos
            non_pos[idx, :2] = sample_nonobj_from_image(lbl, selected, self.method)
            non_pos[idx, 2] = self.num_class

        # shuffle the labeled data
        res = np.vstack([pos, non_pos])
        if self.shuffle:
            np.random.shuffle(res)

        #         images = torch.FloatTensor(res.shape[0], self.window_size, self.window_size).fill_(0)
        images = []
        for idx, label in enumerate(res):
            x, y = label[:2]

            y = 1 - y

            x = int(x * w)
            y = int(y * h)

            window_size = self.window_size
            if self.multiscale:
                window_size = int(np.random.uniform(low=0.6, high=1.5) * window_size)

            box = [max(0, x - math.floor(window_size / 2)), max(0, y - math.floor(window_size / 2)),
                   min(w, x + math.ceil(window_size / 2)), min(h, y + math.ceil(window_size / 2))]

            raw_img = Image.new('RGB', (window_size, window_size)).convert('L')
            raw_img.paste(img.crop(box), (
                math.floor(window_size / 2) - x + box[0],
                math.floor(window_size / 2) - y + box[1]))

            if self.transform:
                raw_img = self.transform(raw_img)
            raw_img = transforms.ToTensor()(raw_img)
            images += [raw_img.unsqueeze(0)]

        return torch.cat(images, dim=0).float(), torch.from_numpy(res[:, 2:]).long()

    def collate_fn(self, batch):
        images, labels = zip(*batch)

        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)

        return images, labels
