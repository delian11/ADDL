import os
import torch
from torchvision.datasets.folder import default_loader

from .datasets_ import *
from .data_transforms import *

def load_data(root, dataset, bs, fold):
    transform_train = data_transform(dataset, 'train')
    transform_test = data_transform(dataset, 'test')
    num_classes = 7
    if dataset == 'RAF':
        train_txt = os.path.join(root, 'RAF/label/train_exp.txt')
        test_txt = os.path.join(root, 'RAF/label/test_exp.txt')
        dset_train = DataSetFromTxt(train_txt, transform=transform_train, root=root + '/RAF/aligned')
        dset_test = DataSetFromTxt(test_txt, transform=transform_test, root=root + '/RAF/aligned')
        print('==> Training dataset: RAF. Total train images: {}, Total validation images: {}'.format(len(dset_train), len(dset_test)))
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=bs, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=bs, shuffle=True, num_workers=2)
    elif dataset =='SFEW':
        train_dir = os.path.join(root, 'SFEW/train_mtcnn/')
        test_dir = os.path.join(root, 'SFEW/val_mtcnn/')
        dset_train = ImageFolder(train_dir, transform = transform_train, loader = default_loader)
        dset_test = ImageFolder(test_dir, transform= transform_test, loader = default_loader)
        print('==> Training dataset: SFEW, Total train images:{}, Total validation images:{}.'.format(len(dset_train), len(dset_test)))
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=bs, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=bs, shuffle=True, num_workers=2)
    else:
        cv_fold = fold
        if dataset == 'CK+':
            data_dir = root + '/CK_128x128_nocrop/cross_validation'+str(cv_fold)
        elif dataset == 'MMI':
            data_dir = root + '/MMI_128x128_nocrop/cross_validation'+str(cv_fold)
            num_classes = 6
        elif dataset == 'OULU':
            data_dir = root + '/OULU_128x128_nocrop/cross_validation'+str(cv_fold)
            num_classes = 6

        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

        dset_train = ImageFolder(train_dir, transform = transform_train, loader = default_loader)
        dset_test = ImageFolder(test_dir, transform= transform_test, loader = default_loader)
        print('==>Training dataset: {}, Total train images:{}, Total validation images:{}.'.format(dataset, len(dset_train), len(dset_test)))
        train_loader = torch.utils.data.DataLoader(dset_train, batch_size=bs, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(dset_test, batch_size=bs, shuffle=True, num_workers=2)
    print('')

    return train_loader, test_loader, num_classes