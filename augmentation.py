import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def horizontal_flip(image, label):
    image = np.flip(image, [0])
    label[:, 1] = 1 - label[:, 1]
    return image, label


def vertical_flip(image, label):
    image = np.flip(image, [1])
    label[:, 0] = 1 - label[:, 0]
    return image, label


def random_crop(image, label):
    size = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9][np.random.randint(0, 6)]
    x, y = np.random.uniform(0, 1-size, 2)
    h, w = image.shape
    # crop img
    image = image[int(y*h):int((y+size)*h), int(x*w):int((x+size)*w)]
    # adjust label
    label = np.array([label[i, :] for i in range(label.shape[0]) if x < label[i, 0] < x + size
                      and y < label[i, 1] < y + size])
    if label.size != 0:
        label[:, 0] = (label[:, 0] - x) / size
        label[:, 1] = (label[:, 1] - y) / size
    return image, label


def random_mask(image, label):
    n = np.random.randint(1, 5)
    h, w = image.shape
    for i in range(n):
        size = [0.05, 0.1, 0.15, 0.2][np.random.randint(0, 3)]
        x, y = np.random.uniform(0, 1-size, 2)
        # mask given area
        image[int(y*h + 0.02):int((y+size)*h - 0.02), int(x*w + 0.02):int((x+size)*w - 0.02)] = np.random.random() * 255
        # adjust label
        label = np.array([label[i, :] for i in range(label.shape[0]) if not (x < label[i, 0] < x + size
                          and y < label[i, 1] < y + size)])
    return image, label


# randomly down sample image and resize back to original size to get images in different resolution
class RandomRescale(object):
    # p = probability of applying this transform
    def __init__(self, p=0.5):
        self.p = p
    # image should be ONE PIL image

    def __call__(self, image):
        if np.random.random() < self.p:
            w, h = image.size
            new_size = np.random.choice(range(int(w / 1.5) - w % 4, w - w % 4, 4))
            image = transforms.Resize(size=(new_size, new_size))(image)
            image = transforms.Resize(size=(w, w))(image)
        return image


# randomly add gaussian noise to the input image
class GaussianNoise(object):
    def __init__(self, p=0.5, std=1):
        self.p = p
        self.std = std

    def __call__(self, image):
        if np.random.random() < self.p:
            image = np.array(image)
            noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
            image = image + noise
            image = Image.fromarray(image).convert('L')
        return image
