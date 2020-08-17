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
parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
parser.add_argument("--log", type=str, default="train_log.txt")
parser.add_argument("--trainset", type=str, required=True, help="path to .csv file specifying train dataset")
parser.add_argument("--testset", type=str, default=None, help="path to .csv file specifying test dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--non_pos_ratio", type=int, default=7, help="non positive ratio in the dataset")
parser.add_argument("--num_class", type=int, default=2, help="number of classes (ignoring background)")
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

model = UnifiedModel(num_class=2, window_size=64)

# if using weights available, load pre-trained weights
if opt.weights is not None:
    print('-- loading weights at {}'.format(opt.weights))
    logging.info('-- loading weights at {}'.format(opt.weights))
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

print('-- loading training set at {}'.format(opt.trainset))
logging.info('-- loading training set at {}'.format(opt.trainset))
dataset = CSVDataset(opt.trainset, transform=transform,
                     method='mixed', non_pos_ratio=opt.non_pos_ratio)

dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
)
print('-- finished loading')

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

lr_train = lambda e: np.power(0.5, int(e / 10))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_train)

weight = torch.FloatTensor([200/597, 397/597, opt.non_pos_ratio]) # ratio for unified_train_hpcc.csv

criterion = nn.CrossEntropyLoss(weight=weight)
criterion.to(device)

if __name__ == '__main__':
    for epoch in range(opt.epochs):
        print("epoch {} | learning rate = {}".format(epoch, "{0:.6f}".format(optimizer.param_groups[0]['lr'])))
        logging.info(
            "epoch {} | learning rate = {}".format(epoch, "{0:.6f}".format(optimizer.param_groups[0]['lr']))
        )
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
            logging.info(
                "epoch {} batch {}, loss = {}".format(epoch, batch_i, loss.item())
            )

        scheduler.step()
        path = 'models/unified_pretrain_{}.pth'.format(epoch)
        print("saving model at {}".format(path))
        logging.info("saving model at {}".format(path))

        torch.save(model.module.state_dict(), path)
        print("model saved\n")

        # test
        if opt.testset is not None and epoch % 5 == 0:

            df_test_labl = pd.read_csv(opt.testset)
            df_test_pred = pd.DataFrame(columns=[
                'path', 'class', 'x', 'y'
            ])

            for path in df_test_labl['path'].unique():
                img = Image.open(path).convert('L')
                w, h = img.size

                img = transforms.ToTensor()(img)
                img = img.to(device)

                f, c = math.ceil(64 / 2) - 1, math.floor(64 / 2)
                img = F.pad(img, (f, c, f, c))

                with torch.no_grad():
                    model.eval()
                    pred = model(img.unsqueeze(0)).squeeze()
                    pred = torch.softmax(pred, dim=0).cpu()

                pred_pos = pred[0, :, :]
                pred_neg = pred[1, :, :]

                pred_pos *= pred.argmax(dim=0) == 0
                pred_neg *= pred.argmax(dim=0) == 1

                pos = peak_local_max(pred_pos.detach().cpu().numpy(), min_distance=12, threshold_abs=0.3,
                                     threshold_rel=0.25).astype(np.float)
                neg = peak_local_max(pred_neg.detach().cpu().numpy(), min_distance=12, threshold_abs=0.3,
                                     threshold_rel=0.25).astype(np.float)

                pos = np.array([[0, y/h, x/w] for x, y in pos])
                neg = np.array([[1, y/h, x/w] for x, y in neg])

                if pos.size == 0:
                    pos = np.array([]).reshape(0, 3)
                if neg.size == 0:
                    neg = np.array([]).reshape(0, 3)

                df_pred = pd.DataFrame(np.vstack((pos, neg)), columns=['class', 'x', 'y'])
                df_pred['path'] = path
                df_test_pred = df_test_pred.append(df_pred, sort=True)

            # convert y to 1 - y for comparison
            df_test_pred['y'] = 1 - df_test_pred['y']
            df_test_pred['detected'] = 0

            print('-- epoch {} | evaluating predictions at {}'.format(epoch, opt.testset))
            logging.info('-- epoch {} | evaluating predictions at {}'.format(epoch, opt.testset))

            dis = evaluate_detection(df_test_labl, df_test_pred)

            print('-- average distance = %.5f' % dis)
            logging.info('-- average distance = %.5f' % dis)
            for mode in ['pos', 'neg', 'total']:
                df_test_pred_cp = df_test_pred.copy()
                df_test_labl_cp = df_test_labl.copy()
                # df_test_pred_cp['detected'] = 0
                # df_test_labl_cp['detected'] = 0

                get_stats(df_test_pred_cp, df_test_labl_cp, mode, log=True)






