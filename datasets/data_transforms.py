import torch
from torchvision.transforms import transforms

def data_transform(dataset, action):
    if dataset in ['SFEW']:
        if action == 'train':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.RandomCrop(90),
                transforms.RandomRotation((-30, 30)),
                transforms.ColorJitter(brightness=(0, 1.5), contrast=0, saturation=0.3, hue=(-0.5, 0.3)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif action == 'test':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.CenterCrop(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    elif dataset in ['RAF']:
        if action == 'train':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.RandomCrop(90),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif action == 'test':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.CenterCrop(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    elif dataset in ['MMI', 'CK+']:
        if action == 'train':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.RandomCrop(90),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif action == 'test':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.CenterCrop(90),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    elif dataset in ['OULU']:
        if action == 'train':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.RandomCrop(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif action == 'test':
            tfs = transforms.Compose([
                transforms.Resize(110),
                transforms.CenterCrop(90),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    
    return tfs