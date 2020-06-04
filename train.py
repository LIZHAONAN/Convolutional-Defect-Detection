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


import numpy as np
import pandas as pd

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
parser.add_argument("--trainset", type=str, required=True, help="path to .csv file specifying train dataset")
parser.add_argument("--testset", type=str, default=None, help="path to .csv file specifying test dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--non_pos_ratio", type=int, default=7, help="non positive ratio in the dataset")
parser.add_argument("--num_class", type=int, default=2, help="number of classes (ignoring background)")
parser.add_argument("--gpu", type=bool, default=True, help="use gpu or not")
parser.add_argument("--weights", type=str, default=None, help="path to pre-trained weights")
opt = parser.parse_args()

if opt.gpu and torch.cuda.is_available():
    print('-- cuda available, using gpu')
    device = torch.device('cuda')
    opt.gpu = True
else:
    print('-- cuda unavailable or unwanted, using cpu')
    device = torch.device('cpu')
    opt.gpu = False

model = UnifiedModel(num_class=2, window_size=64)

# if using weights available, load pre-trained weights
if opt.weights is not None:
    print('-- loading weights at {}'.format(opt.weights))
    model_dict = torch.load(opt.weights, map_location=device)
    model.load_state_dict(model_dict)
    print('-- finished loading')

# if using gpu, wrap models with data parallel
if opt.gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

transform = transforms.Compose([
    RandomRescale(p=0.5),
    GaussianNoise(p=0.5, std=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(size=(64, 64))
])

print('-- loading training set')
dataset = CSVDataset(opt.trainset, transform=transform,
                     method='mixed', non_pos_ratio=opt.non_pos_ratio)

dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
)
print('-- finished loading')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)

weight = torch.ones(opt.num_class + 1)
weight[-1] = opt.non_pos_ratio * 2
criterion = nn.CrossEntropyLoss(weight=weight)
criterion.to(device)

if __name__ == '__main__':
    for epoch in range(opt.epochs):
        for batch_i, (images, labels) in enumerate(dataloader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)

            pred = model(images).squeeze()

            loss = criterion(pred, labels.squeeze())
            loss.backward()
            optimizer.step()

            print("epoch {} batch {}, loss = {}".format(epoch, batch_i, loss.item()))

        path = 'models/unified_pretrain_{}.pth'.format(epoch)
        print("saving model at {}".format(path))
        torch.save(model.module.state_dict(), path)
        print("model saved\n")

