from typing import *
import numpy as np
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data.distributed import DistributedSampler

import os
import os.path
import numpy as np

from svhn.data import SVHN
from cifar10.data import HierarchyCIFAR10
from cifar100.data import HierarchyCIFAR100
from imagenet100.data import ImageNet100, HierarchyImageNet100


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def make_dataloader(exp_name : str, num_workers : int, batch_size : int, 
                    task : str, in_dataset_name : str, 
                    ood_dataset_name : str = None, 
                    ood_root : str = '/path/to/ood_datasets/') -> Tuple[DataLoader, DataLoader]:
    '''
    Creat (a subset of) train test dataloader. Train & test has the same number of classes.
    Args:
        num_workers : number of workers of train and test loader.
        batch_size : batch size of train and test loader
        task : 'train','test', 'ood'
    '''
    
    if in_dataset_name == 'CIFAR10':
        img_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616] 
    elif in_dataset_name == 'CIFAR100':
        img_size = 32
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif in_dataset_name == 'IMAGENET100':
        img_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError()
    normalize = transforms.Normalize(mean=mean, std=std)
    
    def make_train_dataset(exp_name, dataset_name):
        if exp_name == 'SupCon':                                 
            # data augmentations for supcon
            if dataset_name == 'IMAGENET100':
                transform = TwoCropTransform(transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.4, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]))
            else:
                transform = TwoCropTransform(transforms.Compose([
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]))
        elif exp_name == 'ERM':
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor(),
                                            normalize
                                            ])

        if dataset_name == 'CIFAR10':
            dataset = HierarchyCIFAR10(root = './data',
                                        train = True,                         
                                        transform = transform, 
                                        download=False      
                                        )
        elif dataset_name == 'CIFAR100':
            dataset = HierarchyCIFAR100(root = './data',
                                        train = True,                         
                                        transform = transform, 
                                        download=False      
                                        )
        elif dataset_name == 'IMAGENET100':
            dataset = HierarchyImageNet100(root = '/data/common/ImageNet100/ImageNet100',
                                        train = True,                         
                                        transform = transform,
                                        )
        return dataset
    
    def make_test_dataset(dataset_name):
        # inD
        transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        normalize
                                        ])

        if dataset_name == 'CIFAR10':
            dataset = CIFAR10(root = './data',
                            train = False,
                            transform = transform)
        elif dataset_name == 'CIFAR100':
            dataset = CIFAR100(root = './data',
                              train = False,
                              transform = transform)
        elif dataset_name == 'IMAGENET100':
            dataset = ImageNet100(root = '/path/to/ImageNet100',
                                  train = False,
                                  transform = transform)
            
        return dataset
    
    def make_outlier_dataset_large(dataset_name):
        if dataset_name == 'CIFAR10':
            dataset = make_test_dataset('CIFAR10')
        elif dataset_name == 'CIFAR100':
            dataset = make_test_dataset('CIFAR100')
        else: # far-ood

            if dataset_name == 'Places365':
                dataset = ImageFolder(root= os.path.join(ood_root, 'Places'),
                                    transform=transforms.Compose([transforms.Resize(img_size), 
                                    transforms.CenterCrop(img_size), transforms.ToTensor(),normalize]))
            elif dataset_name == 'SUN':
                dataset = ImageFolder(root = os.path.join(ood_root, 'SUN'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))
            elif dataset_name == 'dtd':
                dataset = ImageFolder(root=os.path.join(ood_root, 'dtd', 'images'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size), transforms.ToTensor(),normalize]))
            elif dataset_name == 'iNaturalist':
                dataset = ImageFolder(root = os.path.join(ood_root, 'iNaturalist'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))
            elif dataset_name == 'placesbg':
                dataset = ImageFolder(root = os.path.join(ood_root, 'placesbg'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))

        return dataset
   
    def make_outlier_dataset(dataset_name):
        if dataset_name == 'CIFAR10':
            dataset = make_test_dataset('CIFAR10')
        elif dataset_name == 'CIFAR100':
            dataset = make_test_dataset('CIFAR100')
        else: # far-ood
            if dataset_name == 'SVHN':
                dataset = SVHN(root=os.path.join(ood_root, 'svhn'), split='test',
                                transform=transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),transforms.ToTensor(), normalize]), download=False)
            elif dataset_name == 'Textures':
                dataset = ImageFolder(root=os.path.join(ood_root, 'dtd', 'images'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size), transforms.ToTensor(),normalize]))
            elif dataset_name == 'Places365':
                dataset = ImageFolder(root= os.path.join(ood_root, 'places365'),
                                    transform=transforms.Compose([transforms.Resize(img_size), 
                                    transforms.CenterCrop(img_size), transforms.ToTensor(),normalize]))
            elif dataset_name == 'LSUN':
                dataset = ImageFolder(root = os.path.join(ood_root, 'LSUN'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))
            elif dataset_name == 'iSUN':
                dataset = ImageFolder(root = os.path.join(ood_root, 'iSUN'),
                                            transform=transforms.Compose([transforms.Resize(img_size), 
                                            transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))
            
            if len(dataset) > 10000: 
                print("Sampling 10000 samples")
                dataset = Subset(dataset, np.random.choice(len(dataset), 10000, replace=False))
        
        return dataset

    if task == 'train':
        dataset = make_train_dataset(exp_name, in_dataset_name)
        world_size = int(os.environ.get('WORLD_SIZE',0))
        if world_size > 1: # DDP
            rank = int(os.environ['RANK'])
            sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    elif task == 'test':
        dataset = make_test_dataset(in_dataset_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif task == 'ood':
        if in_dataset_name == 'IMAGENET100':
            dataset = make_outlier_dataset_large(ood_dataset_name)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            dataset = make_outlier_dataset(ood_dataset_name)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader
