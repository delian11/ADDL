# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, default_loader
import sys, os
import csv
import scipy.io as sio
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

class DataSetFromTxt(Dataset):
    def __init__(self, txtfile, transform=None, target_transform=None, loader=default_loader, root='/home/ruan/DATA/RAF'):
        super(DataSetFromTxt, self).__init__()
        file = open(txtfile, 'r')
        imgs = []
        targets = []
        for line in file:
            line = line.strip('\n')
            words = line.split(' ')
            imgs.append(words[0])
            targets.append(words[1:])

        self.imgs = imgs
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.root = root
        file.close()

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        imgpath = os.path.join(self.root, imgpath)
        img_PIL = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img_PIL)

        targets = self.targets[index]
        # print(len(targets))
        targets = list(map(eval, targets))
        targets = np.array(targets)
        targets = torch.LongTensor(targets)
        
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        targets = targets.squeeze()
        return img, targets

    def __len__(self):
        return len(self.imgs)


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS) #
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        def get_item(index):
            path, target = self.samples[index]
            # =============================================================================
            #             while target == 0:
            #                 index = np.random.randint(0, len(self.samples))
            #                 path, target = self.samples[index]
            # =============================================================================
            sample = self.loader(path)
            
            sample_gray = F.to_grayscale(sample)
            img_array = np.array(sample_gray,dtype=np.uint8)
            h,w, = np.shape(img_array)
        
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        
        sample, target = get_item(index)
        return sample, target

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # class_to_idx = {classes[0]: 0 }
        return classes, class_to_idx
    
    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

