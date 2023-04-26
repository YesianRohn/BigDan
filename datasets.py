# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data.sampler import BatchSampler, Sampler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.datasets.vision import VisionDataset

import torch.utils.data as data
import utils
import itertools

class MultiImageFolder(data.Dataset):
    def __init__(self, dataset_list, transform, loader = default_loader, 
                    known_data_source=True, is_test=False) -> None:
        super().__init__()
        self.loader = loader
        self.transform = transform

        samples_list = [x.samples for x in dataset_list]
        classes_list = [x.classes for x in dataset_list]
        self.classes_list = classes_list
        self.dataset_list = dataset_list
        self.classes = [y for x in self.classes_list for y in x]

        start_id = 0
        self.samples = []
        for dataset_id, (samples, classes) in enumerate(zip(samples_list, classes_list)) :
            for i, data in enumerate(samples):
                if not is_test:
                    # concat the taxonomy of all datasets
                    img, target = data[:2]
                    self.samples.append((img, target+start_id, dataset_id))
                    samples[i] = (img, target+start_id)
                else :
                    img = data
                    self.samples.append((img, None, dataset_id))
            start_id += len(classes)

    def __len__(self, ):
        return len(self.samples)

    def __getitem__(self, index) :
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        path, target, dataset_id = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, dataset_id


class TestFolder(data.Dataset):
    def __init__(self, image_root, transform, loader = default_loader):
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.classes = os.listdir(os.path.join(image_root, 'train'))
        image_root = os.path.join(image_root, 'test')

        for file_name in os.listdir(image_root):
            self.samples.append(os.path.join(image_root, file_name))

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index) :
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        path = self.samples[index]
        target = None
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, image_id


def build_dataset(is_train, args):
    is_test = not is_train and args.test_only
    transform = build_transform(is_train, args)

    dataset_list = []
    nb_classes = 0

    if is_test:
        for dataset in args.dataset_list :
            root = os.path.join(args.data_path, dataset)
            dataset = TestFolder(root, transform=transform)
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)
        multi_dataset = MultiImageFolder(dataset_list, transform, is_test=True)
        return multi_dataset, nb_classes, None
    else :
        for dataset in args.dataset_list :
            root = os.path.join(args.data_path, dataset, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)

        multi_dataset = MultiImageFolder(dataset_list, transform, known_data_source=args.known_data_source)

        return multi_dataset, nb_classes


def build_transform(is_train, args, img_size=224,
                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    # TODO: does any other data augmentation work better?
    if is_train:
        t = []
        t.append(transforms.Resize(img_size))
        t.append(transforms.CenterCrop(img_size))
        if args.flip:
            t.append(transforms.RandomVerticalFlip(p = args.flip))
            t.append(transforms.RandomHorizontalFlip(p = args.flip))
        if args.rotation:
            t.append(transforms.RandomRotation(args.rotation))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

    t = []
    t.append(transforms.Resize(img_size))
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class GroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_sizes, num_datasets):
        """
        Group images from the same dataset into a batch
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(num_datasets)]

    def __len__(self):
        return len(self.dataset) // self.batch_sizes

    def __iter__(self):
        iter_id = 0
        while True :
            for d in self.dataset:
                image, target, dataset_id = d
                bucket = self._buckets[dataset_id]
                bucket.append(d)
                if len(bucket) == self.batch_sizes:
                    images, targets, dataset_ids = list(zip(*bucket))
                    images = torch.stack(images)
                    targets = torch.tensor(targets)
                    dataset_ids = torch.tensor(dataset_ids)
                    del bucket[:]
                    yield images, targets, dataset_ids
                    iter_id +=1
                    if iter_id == len(self):
                        return
