from model import *
from utils import detect_image
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from skimage.feature import peak_local_max

from PIL import Image

import argparse
import math
import numpy as np
import pandas as pd

import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="path to the weight of the model")
parser.add_argument("--images", type=str, required=True, help="path to image path file in .csv format")
parser.add_argument("--output", type=str, required=True, help="path to the output file")
parser.add_argument("--conf", type=str, default=None, help='folder to save intermediate confidence maps')
parser.add_argument("--window", type=int, default=64, help='window size of detector')


opt = parser.parse_args()

device = torch.device('cuda')

print('-- loading models')
model = UnifiedModel(num_class=2, window_size=64)
model.load_state_dict(torch.load('models/unified_99.pth'))
model = model.cuda()
# model = torch.nn.DataParallel(model)

model = model.to(device)
model.eval()
print('-- finished loading\n')

# load image paths
df_img_path = pd.read_csv(opt.images, sep=',')

# create result file
df_img_res = pd.DataFrame(columns=['path', 'class', 'x', 'y'])


if __name__ == '__main__':
    for idx, row in df_img_path.iterrows():
        path = row['path']
        file_name = os.path.basename(path)

        print('\n-- evaluating {} ...'.format(path))
        since = time.time()

        img = Image.open(path).convert('L')
        w, h = img.size

        if max(w, h) > 400:
            new_w, new_h = int(w * 400 / max(w, h)), int(h * 400 / max(w, h))
            print('-- image too big ({}, {}), resize image to ({}, {}) for detection'.format(w, h, new_w, new_h))
            w, h = new_w, new_h

        img = transforms.Resize(size=(w, h))(img)
        img = transforms.ToTensor()(img)
        img = img.to(device)

        f, c = math.ceil(opt.window / 2) - 1, math.floor(opt.window / 2)
        img = F.pad(img, (f, c, f, c))

        # pred = model(img.unsqueeze(0)).squeeze()
        pred = detect_image(model, img, section_size=400)
        pred = torch.softmax(pred, dim=0).cpu()

        pred_pos = pred[0, :, :]
        pred_neg = pred[1, :, :]

        pred_pos *= pred.argmax(dim=0) == 0
        pred_neg *= pred.argmax(dim=0) == 1

        pos = peak_local_max(pred_pos.detach().cpu().numpy(), min_distance=12, threshold_abs=0.3,
                             threshold_rel=0.25).astype(np.float)
        neg = peak_local_max(pred_neg.detach().cpu().numpy(), min_distance=12, threshold_abs=0.3,
                             threshold_rel=0.25).astype(np.float)

        pos = np.array([[0, x / w, y / h] for x, y in pos])
        neg = np.array([[1, x / w, y / h] for x, y in neg])

        if pos.size == 0:
            pos = np.array([]).reshape(0, 3)
        if neg.size == 0:
            neg = np.array([]).reshape(0, 3)

        time_used = time.time() - since
        print('-- total {} positive defects and {} negative defects detected'.format(len(pos), len(neg)))
        print('-- finished {} in {:.0f}m {:.0f}s'.format(path, time_used // 60, time_used % 60))

        df_img = pd.DataFrame(np.vstack((pos, neg)), columns=['class', 'x', 'y'])
        df_img['path'] = path

        df_img_res = df_img_res.append(df_img, sort=True)

        df_img_res.to_csv(opt.output, sep=',', index=None)
        print('-- defect result stored at {}'.format(opt.output))

        if opt.conf:
            print('-- saving confidence map...')
            pred = pred.detach().cpu().numpy()
            with open(os.path.join(opt.conf, 'conf_' + file_name), 'wb') as f:
                np.save(f, pred)
            print('-- confidence map saved at {}'.format(os.path.join(opt.conf, 'conf_' + file_name)))
