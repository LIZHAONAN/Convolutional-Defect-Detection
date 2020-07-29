import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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


class RecurrentCSVDataset(Dataset):
    def __init__(self, path_to_csv, window_size=64, non_pos_ratio=7, pos_per_image=8, num_class=2,
                 method='tracking', multiscale=True, transform=True, shuffle=True, start_idx=None):
        assert (method in ['tracking', 'detection'])
        self.df = pd.read_csv(path_to_csv, sep=',')

        #         self.df['path'] = [path.replace('3_Low_Contrast_Adjusted', '3_Low_Contrast_Adjusted_Resized')
        #                            for path in self.df['path']]

        self.img_paths = self.df['path'].unique()

        self.window_size = window_size
        self.non_pos_ratio = non_pos_ratio
        self.pos_per_image = pos_per_image
        self.num_class = num_class
        self.method = method

        self.transform = transform
        self.multiscale = multiscale
        self.shuffle = shuffle
        self.start_idx = start_idx

    def __len__(self):
        if self.method == 'tracking':
            return len(self.img_paths) - 1
        return len(self.img_paths)

    # apply the same transform to current and previous samples
    def transform_img(self, image, prev=None, conf=None, prob=None):
        if not prob:
            prob = np.random.random(7)

        # random scale:
        if prob[0] < 0.5:
            w, h = image.size
            new_size = np.random.choice(range(int(w / 1.5) - w % 4, w - w % 4, 4))
            image = transforms.Resize(size=(new_size, new_size))(image)
            image = transforms.Resize(size=(w, h))(image)
            if prev:
                prev = transforms.Resize(size=(new_size, new_size))(prev)
                prev = transforms.Resize(size=(w, h))(prev)

        # Gaussian Noise
        if prob[1] < 0.5:
            image = np.array(image)
            noise = np.random.normal(loc=0, scale=10, size=image.shape)
            image = image + noise
            image = Image.fromarray(image).convert('L')
            if prev:
                prev = np.array(prev)
                prev = prev + noise
                prev = Image.fromarray(prev).convert('L')

        # Brightness, contrast and saturation change
        if prob[2] < 0.5:
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(image)
            if prev:
                prev = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(prev)

        # Random Rotation
        if prob[3] < 0.5:
            angle = random.randint(-15, 15)
            image = TF.rotate(image, angle)
            if prev:
                prev = TF.rotate(prev, angle)
                conf = TF.rotate(conf, angle)

        # Vertical flip
        if prob[5] < 0.5:
            image = TF.vflip(image)
            if prev:
                prev = TF.vflip(prev)
                conf = TF.vflip(conf)

        # Resize to given window size
        image = TF.resize(image, size=(self.window_size, self.window_size))
        if prev:
            prev = TF.resize(prev, size=(self.window_size, self.window_size))
            return image, prev, conf

        else:
            return image

    def to_tensor(self, img):
        return transforms.ToTensor()(img)

    def get_img_sample(self, img, x, y, w, h, tracking=False):
        window_size = self.window_size
        box = [max(0, x - math.floor(window_size / 2)), max(0, y - math.floor(window_size / 2)),
               min(w, x + math.ceil(window_size / 2)), min(h, y + math.ceil(window_size / 2))]

        raw_img = Image.new('RGB', (window_size, window_size)).convert('L')
        raw_img.paste(img.crop(box), (
            math.floor(window_size / 2) - x + box[0],
            math.floor(window_size / 2) - y + box[1]))

        return raw_img

    # sample detection label
    def get_detection_sample(self, index):
        # img_path = self.img_paths[index]
        # w, h = img.size
        #
        # lbl = self.df[self.df['path'] == img_path]
        # lbl = lbl[['x', 'y', 'class']].values
        #
        # pos_mask = np.random.choice(lbl.shape[0], self.pos_per_image)
        # pos = np.zeros((self.pos_per_image, 3))  # [x, y, classes]
        # pos[:, :2] = lbl[pos_mask, :2]
        # pos[:, 2] = lbl[pos_mask, 2]
        #
        # non_pos = np.zeros((self.non_pos_ratio * self.pos_per_image, 3))
        # for idx, _ in enumerate(non_pos):
        #     selected = pos
        #     non_pos[idx, :2] = sample_nonobj_from_image(lbl, selected, method='mixed')
        #     non_pos[idx, 2] = self.num_class
        #
        # res = np.vstack([pos, non_pos])
        # if self.shuffle:
        #     np.random.shuffle(res)
        #
        # images = []
        # for idx, label in enumerate(res):
        #     x, y = label[:2]
        #
        #     y = 1 - y
        #
        #     x = int(x * w)
        #     y = int(y * h)
        #
        #     window_size = self.window_size
        #     if self.multiscale:
        #         window_size = int(np.random.uniform(low=0.6, high=1.5) * window_size)
        #
        #     raw_img = self.get_img_sample(img, x, y, w, h, window_size)
        #     raw_img = self.transform_img(raw_img)
        #     raw_img = self.to_tensor(raw_img)
        #
        #     images += [raw_img.unsqueeze(0)]
        #
        # return torch.cat(images, dim=0).float(), torch.from_numpy(res[:, 2:]).long()
        return None

    def get_tracking_sample(self, index):
        frst_path = self.img_paths[index]

        # to improve robustness, the current frame is randomly sampled from 1 to 2 frames after
        # prev frame
        if index < len(self.img_paths) - 3 and np.random.random() < 0.5:
            offset = np.random.choice([2, 3], 1)
            scnd_path = self.img_paths[index + offset[0]]
        else:
            scnd_path = self.img_paths[index + 1]
        frst_img = Image.open(frst_path).convert('L')
        scnd_img = Image.open(scnd_path).convert('L')
        w, h = frst_img.size

        frst_labl = self.df[self.df['path'] == frst_path][['x', 'y', 'class']].values
        scnd_labl = self.df[self.df['path'] == scnd_path][['x', 'y', 'class']].values

        res = self.get_tracking_labels(frst_labl, scnd_labl)

        frst_samples, scnd_samples, conf_maps = [], [], []
        for idx, label in enumerate(res):
            u, v = label[:2]
            x = u
            y = 1 - v

            x = int(x * w)
            y = int(y * h)

            window_size = self.window_size
            if self.multiscale:
                window_size = int(np.random.uniform(low=0.6, high=1.5) * window_size)

            img1 = self.get_img_sample(frst_img, x, y, w, h, window_size)
            img2 = self.get_img_sample(scnd_img, x, y, w, h, window_size)
            conf_map = self.get_conf_map(frst_labl, u, v, w, h, window_size)
            conf_map = Image.fromarray(conf_map.squeeze().numpy())

            if self.transform:
                img1, img2, conf_map = self.transform_img(img1, img2, conf_map)
            img1, img2, conf_map = self.to_tensor(img1), self.to_tensor(img2), self.to_tensor(conf_map)

            frst_samples += [img1.unsqueeze(0)]
            scnd_samples += [img2.unsqueeze(0)]
            conf_maps += [conf_map.unsqueeze(0)]

        frst_samples = torch.cat(frst_samples, dim=0).float()
        scnd_samples = torch.cat(scnd_samples, dim=0).float()
        conf_maps = torch.cat(conf_maps, dim=0).float()

        return torch.cat((frst_samples, scnd_samples, conf_maps), dim=1), torch.from_numpy(res[:, 2:]).long()

    def get_conf_map(self, labl, x, y, w, h, window_size):
        rw = (window_size / w) / 2
        rh = (window_size / h) / 2
        pts = labl[(x - rw < labl[:, 0]) & (y - rh < labl[:, 1]) &
                   (x + rw > labl[:, 0]) & (y + rh > labl[:, 1]), ...]
        pts[:, 0] = (pts[:, 0] - (x - rw)) / (rw * 2)
        pts[:, 1] = (pts[:, 1] - (y - rh)) / (rh * 2)
        pts[:, 1] = 1 - pts[:, 1]
        #         pts[:, :2] = pts[:, [1, 0]]
        return self.gaussian_rendering(pts)

    def gaussian_rendering(self, pts, window_size=64, theta=torch.tensor(0.025)):
        W = window_size
        heat_map = torch.zeros((1, window_size, window_size))

        if pts.size > 0:
            temp = torch.zeros((pts.shape[0], window_size, window_size))
            for idx, row in enumerate(pts):
                x_offset = torch.from_numpy(np.arange(W)).expand((W, W)).float() / W
                y_offset = torch.from_numpy(np.arange(W)).unsqueeze(0).t().expand((W, W)).float() / W
                x_offset = x_offset.unsqueeze(0)
                y_offset = y_offset.unsqueeze(0)
                dis = torch.cat((
                    x_offset - row[0],
                    y_offset - row[1]
                ), dim=0)
                temp[idx, ...] = torch.sum(torch.pow(dis, 2), dim=0)
            temp = torch.exp(-temp / (2 * torch.pow(theta, 2)))
            heat_map = temp.max(dim=0).values
            heat_map = heat_map.unsqueeze(0)
        return heat_map

    def get_tracking_labels(self, frst_labl, scnd_labl):
        pos = np.zeros((self.pos_per_image, 3))  # [x, y, cls]
        non_pos = np.zeros((self.pos_per_image * self.non_pos_ratio, 3))

        pos_mask = np.random.choice(scnd_labl.shape[0], self.pos_per_image)
        pos[:, :2] = scnd_labl[pos_mask, :2]
        pos[:, 2] = scnd_labl[pos_mask, 2]

        for idx, _ in enumerate(non_pos):
            non_pos_x, non_pos_y = sample_nonobj_from_image(scnd_labl, pos, method='mixed')
            non_pos[idx, :] = [non_pos_x, non_pos_y, self.num_class]

        res = np.vstack([pos, non_pos])
        if self.shuffle:
            np.random.shuffle(res)

        return res

    def __getitem__(self, index):
        if self.method == 'detection':
            return self.get_detection_sample(index)
        elif self.method == 'tracking':
            if self.start_idx is not None:
                index = self.start_idx[index % len(self.start_idx)]
            return self.get_tracking_sample(index)
        else:
            print('method not supported')
            return None

    def collate_fn(self, batch):
        images, labels = zip(*batch)

        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)

        return images, labels