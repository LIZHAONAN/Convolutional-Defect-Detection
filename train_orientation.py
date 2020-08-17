from model import *
from dataset import *
from augmentation import *
from utils import *

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from skimage.feature import peak_local_max

import logging

import numpy as np
import pandas as pd

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--log", type=str, default="train_log.txt")
parser.add_argument("--trainset", type=str, default=None, help="path to .csv file specifying train dataset")
parser.add_argument("--testset", type=str, default=None, help="path to .csv file specifying test dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--gpu", type=bool, default=True, help="use gpu or not")
parser.add_argument("--weights", type=str, default=None, help="path to pre-trained weights")
opt = parser.parse_args()

logging.basicConfig(filename=opt.log, filemode='w', level=logging.INFO)

if opt.gpu and torch.cuda.is_available():
    print('-- cuda available, using gpu')
    logging.info("-- cuda available, using gpu")
    device = torch.device('cuda')
    opt.gpu = True
else:
    print('-- cuda unavailable or unwanted, using cpu')
    logging.info('-- cuda unavailable or unwanted, using cpu')
    device = torch.device('cpu')
    opt.gpu = False

model = OrientationModel(num_dim=3, window_size=64)
model.load_from_weights(opt.weights)

# if using weights available, load pre-trained weights
# if opt.weights is not None:
    # print('-- loading weights at {}'.format(opt.weights))
    # logging.info('-- loading weights at {}'.format(opt.weights))
    # model_dict = torch.load(opt.weights, map_location=device)
    # model.load_state_dict(model_dict)
    # print('-- finished loading')

# if using gpu, wrap models with data parallel
if opt.gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

transform = transforms.Compose([
    RandomRescale(p=0.5),
    GaussianNoise(p=0.5, std=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(size=(64, 64))
])


if opt.trainset is None:
    print('-- loading orientation pretrain data set')
    logging.info('-- loading orientation pretrain data set')
    dataset = OrientationPretrainDataset(window_size=64, length=64*100, transform=transform)
else:
    print('-- loading training set at {}'.format(opt.trainset))
    logging.info('-- loading training set at {}'.format(opt.trainset))
    dataset = OrientationDataset(opt.trainset, transform=transform)

dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True
)
print('-- finished loading')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)

lr_train = lambda e: np.power(0.5, int(e / 50))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_train)

# criterion = nn.MSELoss(reduction='sum')
# criterion.to(device)

pi = torch.acos(torch.zeros(1)).item() * 2


def Loss(pred, labl, type='pos'):
    diff = labl - pred
    if type == 'pos':
        loss = 1 - torch.abs(torch.cos(diff / 2.))
        return torch.sum(loss)
    elif type == 'neg':
        loss = 1 - torch.abs(torch.cos(diff / 2. * 3.))
        return torch.sum(loss)


if __name__ == '__main__':
    for epoch in range(opt.epochs):
        print("epoch {} | learning rate = {}".format(epoch, "{0:.7f}".format(optimizer.param_groups[0]['lr'])))
        logging.info(
            "epoch {} | learning rate = {}".format(epoch, "{0:.7f}".format(optimizer.param_groups[0]['lr']))
        )
        for batch_i, (images, labels) in enumerate(dataloader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            cls = labels[:, 0].long()
            theta = labels[:, 1].float()

            pred = model(images).squeeze()

            pos_mask = cls == 0
            neg_mask = cls == 1

            pos_loss = Loss(pred[pos_mask], theta[pos_mask], type='pos') if torch.any(
                pos_mask) else torch.zeros(1)
            neg_loss = Loss(pred[neg_mask], theta[neg_mask], type='neg') if torch.any(
                neg_mask) else torch.zeros(1)
            loss = pos_loss + neg_loss
            loss /= torch.sum(pos_mask + neg_mask)
            loss.backward()

            optimizer.step()

            print("epoch {} batch {}, loss = {}".format(epoch, batch_i, loss.item()))
            logging.info(
                "epoch {} batch {}, loss = {}".format(epoch, batch_i, loss.item())
            )

        scheduler.step()
        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            path = 'models/orientation_pretrain_{}.pth'.format(epoch)
            print("saving model at {}".format(path))
            logging.info("saving model at {}".format(path))

            torch.save(model.module.state_dict(), path)
            print("model saved\n")

        # test
        if opt.testset is not None and (epoch % 5 == 0 or epoch == opt.epochs - 1):

            df_test_labl = pd.read_csv(opt.testset)
            df_test_labl['pred'] = -1.
            df_test_labl['diff'] = -1.

            for i, row in df_test_labl.iterrows():
                path, x, y, theta, cls = row[['path', 'x', 'y', 'theta', 'class']]
                img = Image.open(path).convert('L')
                w, h = img.size

                y = 1 - y

                x = int(x * w)
                y = int(y * h)

                window_size = 64

                box = [max(0, x - math.floor(window_size / 2)), max(0, y - math.floor(window_size / 2)),
                       min(w, x + math.ceil(window_size / 2)), min(h, y + math.ceil(window_size / 2))]

                raw_img = Image.new('RGB', (window_size, window_size)).convert('L')
                raw_img.paste(img.crop(box), (
                    math.floor(window_size / 2) - x + box[0],
                    math.floor(window_size / 2) - y + box[1]))

                img = transforms.ToTensor()(raw_img)

                with torch.no_grad():
                    model.eval()
                    pred = model(img.unsqueeze(0)).squeeze().cpu().numpy()
                    pred = pred % (2 * np.pi)

                    if cls == 0:
                        diff = abs(pred - theta)
                        diff = min(diff, abs(2*np.pi - diff))
                    elif cls == 1:
                        diff = abs(pred - theta)
                        diff = min(diff, abs(2*np.pi/3 - diff))

                    df_test_labl.at[i, 'pred'] = pred
                    df_test_labl.at[i, 'diff'] = diff

            diff = df_test_labl['diff'].values
            error = np.mean(diff)
            print('number of test defects = {}, average difference = {:.4f} or {:.2f} degree'.format(
                diff.shape[0], error, error / np.pi * 180
            ))
            logging.info('number of test defects = {}, average difference = {:.4f} or {:.2f} degree'.format(
                diff.shape[0], error, error / np.pi * 180
            ))






