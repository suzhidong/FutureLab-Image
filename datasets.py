import os
import cv2
import numpy
import random
import torch
from cutout import Cutout
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data
from disturb_label import create_disturb_label


def get_train_loader(args):
    dataset = HCCRTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


def get_test_loader(args):
    dataset = HCCRTestSet(args)

    return DataLoader(
        dataset,
        batch_size=1 if args.model == 'nasnet' or args.size > 224 else args.batch_size // 2,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)


class HCCRTrainSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()
        self.args = args

        # for path, _, image_set in os.walk(os.path.join(args.data_dir, 'train')):
        #     if os.path.isdir(path):
        #         for image in image_set:
        lines = open(args.train_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(args.data_dir, path))
            self.targets.append(int(label))

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev),

        ])

        if args.cutout:
            self.transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.args.size, self.args.size))

        image = self.transform(image)

        return image, self.targets[index]

    def __len__(self):
        return len(self.targets)


class HCCRTestSet(data.Dataset):
    def __init__(self, args):
        self.images = list()
        self.targets = list()
        self.args = args

        lines = open(args.test_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(args.data_dir, path))
            self.targets.append(int(label))

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.args.size, self.args.size))

        image = self.transform(image)

        target = self.targets[index]

        if (self.args.enable_disturb_label):
            target = create_disturb_label(target, self.args.noise_rate)

        return image, target, self.images[index]

    def __len__(self):
        return len(self.targets)
