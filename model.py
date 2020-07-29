# Darknet based one pass defect detection model
# Author: Zhaonan Li, zli@brandeis.edu
# Created at 5/12/2020

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import *
from block import *


# convolution with batch norm
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, pad, stride=1, activation=True, attn=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, stride=stride, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm2d(num_features=out_dim)
        ]
        self.op = nn.Sequential(*layers)
        self.relu = nn.ReLU() if activation else None
        self.attn = AttnBlock(in_dim=out_dim) if attn else None

    def forward(self, x):
        out = self.op(x)
        if self.relu is not None:
            out = self.relu(out)
        if self.attn is not None:
            out = self.attn(out)
        return out


# residual layer
class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_sizes=(1, 3), attn=False):
        super(ResLayer, self).__init__()
        k1, k2 = kernel_sizes
        self.conv1 = ConvBlock(in_dim=in_dim, out_dim=out_dim, stride=1, kernel_size=k1, pad=k1//2, attn=attn)
        self.conv2 = ConvBlock(in_dim=out_dim, out_dim=out_dim, stride=1, kernel_size=k2, pad=k2//2,
                               activation=False, attn=attn)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(residual + out)
        return out


# residual block, consisting of many residual layers
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_sizes=(1, 3), num_layers=1, attn=False):
        super(ResBlock, self).__init__()
        layers = [ResLayer(in_dim, out_dim, kernel_sizes, attn=attn)] * num_layers
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class UnifiedModel(nn.Module):
    def __init__(self, num_class, window_size=64):
        super(UnifiedModel, self).__init__()
        self.window_size = window_size
        self.num_class = num_class

        encoder_list = [
            ConvBlock(in_dim=1, out_dim=32, kernel_size=3, pad=1),
            ResBlock(in_dim=32, out_dim=32, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=32, out_dim=64, kernel_size=3, pad=1),
            ResBlock(in_dim=64, out_dim=64, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=64, out_dim=128, kernel_size=3, pad=1),
            ResBlock(in_dim=128, out_dim=128, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=128, out_dim=256, kernel_size=3, pad=1),
            ResBlock(in_dim=256, out_dim=256, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=256, out_dim=512, kernel_size=3, pad=1)
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = ConvBlock(in_dim=512, out_dim = num_class + 1, kernel_size=window_size, pad=0)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class UnifiedModelAttn(nn.Module):
    def __init__(self, num_class, window_size=64):
        super(UnifiedModelAttn, self).__init__()
        self.window_size = window_size
        self.num_class = num_class

        encoder_list = [
            ConvBlock(in_dim=1, out_dim=32, kernel_size=3, pad=1, attn=True),
            ResBlock(in_dim=32, out_dim=32, kernel_sizes=(1, 3), num_layers=1, attn=True),
            ConvBlock(in_dim=32, out_dim=64, kernel_size=3, pad=1, attn=True),
            ResBlock(in_dim=64, out_dim=64, kernel_sizes=(1, 3), num_layers=1, attn=True),
            ConvBlock(in_dim=64, out_dim=128, kernel_size=3, pad=1, attn=True),
            ResBlock(in_dim=128, out_dim=128, kernel_sizes=(1, 3), num_layers=1, attn=True),
            ConvBlock(in_dim=128, out_dim=256, kernel_size=3, pad=1, attn=True),
            ResBlock(in_dim=256, out_dim=256, kernel_sizes=(1, 3), num_layers=1, attn=True),
            ConvBlock(in_dim=256, out_dim=512, kernel_size=3, pad=1, attn=True)
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = ConvBlock(in_dim=512, out_dim = num_class + 1, kernel_size=window_size, pad=0, attn=False)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out



class UnifiedModelTiny(nn.Module):
    def __init__(self, num_class, window_size=45):
        super(UnifiedModelTiny, self).__init__()
        self.window_size = window_size
        self.num_class = num_class

        encoder_list = [
            ConvBlock(in_dim=1, out_dim=32, kernel_size=3, pad=1),
            ResBlock(in_dim=32, out_dim=32, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=32, out_dim=64, kernel_size=3, pad=1),
            ResBlock(in_dim=64, out_dim=64, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=64, out_dim=128, kernel_size=3, pad=1),
            ResBlock(in_dim=128, out_dim=128, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=128, out_dim=256, kernel_size=3, pad=1),
            ResBlock(in_dim=256, out_dim=256, kernel_sizes=(1, 3), num_layers=1),
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = ConvBlock(in_dim=256, out_dim = num_class + 1, kernel_size=window_size, pad=0)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class TrackingConditionedModel(nn.Module):
    def __init__(self, num_class, window_size=64):
        super(TrackingConditionedModel, self).__init__()
        self.window_size = window_size
        self.num_class = num_class

        encoder_list = [
            ConvBlock(in_dim=3, out_dim=32, kernel_size=3, pad=1),
            ResBlock(in_dim=32, out_dim=32, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=32, out_dim=64, kernel_size=3, pad=1),
            ResBlock(in_dim=64, out_dim=64, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=64, out_dim=128, kernel_size=3, pad=1),
            ResBlock(in_dim=128, out_dim=128, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=128, out_dim=256, kernel_size=3, pad=1),
            ResBlock(in_dim=256, out_dim=256, kernel_sizes=(1, 3), num_layers=1),
            ConvBlock(in_dim=256, out_dim=512, kernel_size=3, pad=1),
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = ConvBlock(in_dim=512, out_dim=num_class + 1, kernel_size=window_size, pad=0)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    # initialize weights from detection model instead of randomly initialize
    def load_from_weights(self, path):
        print('-- loading detection model weights from {}'.format(path))
        own_state = self.state_dict()
        det_state = torch.load(path)
        for name in det_state:
            param = det_state[name]
            if name not in own_state:
                print('skipping {} in the detection model'.format(name))
                continue
            if name == 'encoder.0.op.0.weight':
                own_state[name][:, 0, :, :] = param.squeeze().clone()
                continue
            own_state[name].copy_(param)
        print('-- finished loading')
