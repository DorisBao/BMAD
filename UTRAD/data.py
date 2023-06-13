import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils import data
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import pandas as pd
from torch.utils.data import DataLoader
import glob

def get_train_transforms():
    r'''
    resize (224) -> tensor -> normalize
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_valid_transforms():
    r'''
    resize (224) -> tensor -> normalize
    '''
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


class TrainDataset(data.Dataset):
    r""" 
    RESC: 4,297 samples
    OCT2017: 26,315 samples
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

        if self.data == 'RESC':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/RESC/Train/train/good/'
        elif self.data == 'OCT2017':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/OCT2017/train/good/'
        elif self.data == 'liver':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/train/good/'
        elif self.data == 'bras2021':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/Brain/train/good/'
        elif self.data == 'chest':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/chest-rsna/Chest-RSNA/train/good/'
        elif self.data == 'camelyon':
            self.data_root = '/home/jinan/Datasets/Medical-datasets/camelyon16_256/train/good/'
        self.data_list = os.listdir(self.data_root)
                
    def load_image(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        img_path = self.data_root + self.data_list[idx]
        img = self.load_image(img_path)
        sample = {'image': img}

        return sample

    def __len__(self):
       return len(self.data_list)

class ValidDataset(data.Dataset):
    r""" 
    RESC: 115 samples (45 normal, 70 abnormal)
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

        if self.data == 'RESC':
            self.good = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val/Ungood/'
            self.mask = '/home/jinan/Datasets/Medical-datasets/RESC/Val/val_label/Ungood/'
        elif self.data == 'OCT2017':
            self.good = '/home/jinan/Datasets/Medical-datasets/OCT2017/val/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/OCT2017/val/Ungood/'
        elif self.data == 'liver':
            self.good = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/img/Ungood/'
            self.mask = '/home/jinan/Datasets/Medical-datasets/Liver/Train/hist_DIY/valid/label/Ungood/'
        elif self.data == 'bras2021':
            self.good = '/home/jinan/Datasets/Medical-datasets/Brain/valid/good/img/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/Brain/valid/Ungood/img/'
            self.mask = '/home/jinan/Datasets/Medical-datasets/Brain/valid/Ungood/label/'
        elif self.data == 'chest':
            self.good = '/home/jinan/Datasets/Medical-datasets/chest-rsna/Chest-RSNA/val/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/chest-rsna/Chest-RSNA/val/Ungood/'
        elif self.data == 'camelyon':
            self.good = '/home/jinan/Datasets/Medical-datasets/camelyon16_256/valid/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/camelyon16_256/valid/Ungood/'
        self.good_list = os.listdir(self.good)
        self.ungood_list = os.listdir(self.ungood)
                
    def load_image(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if idx < len(self.ungood_list):
            img_path = self.ungood + self.ungood_list[idx]
            img = self.load_image(img_path)
            if self.data == 'OCT2017' or self.data == 'chest' or self.data == 'camelyon':
                mask = torch.zeros(1,256, 256)
            else:
                mask_path = self.mask + self.ungood_list[idx]
                mask_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                mask = mask_transform(Image.open(mask_path).convert('L'))
                mask[mask>=0.1] = 1
                mask[mask<0.1] = 0
            label = 1
        else:
            img_path = self.good + self.good_list[idx-len(self.ungood_list)]
            img = self.load_image(img_path)
            mask = torch.zeros(1,256, 256)
            label = 0
        
        sample = {'image': img, 'mask': mask, 'label': label, 'path': str(img_path)}

        return sample

    def __len__(self):
       return len(self.good_list) + len(self.ungood_list)


class TestDataset(data.Dataset):
    r""" 
    RESC: 1805 samples (1041 normal, 764 abnormal)
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

        if self.data == 'RESC':
            self.good = '/home/jinan/Datasets/gragh_resc/'
            self.ungood = '/home/jinan/Datasets/gragh_resc/'
            self.mask = '/home/jinan/Datasets/gragh_resc/'
        elif self.data == 'OCT2017':
            self.good = '/home/jinan/Datasets/Medical-datasets/OCT2017/test/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/OCT2017/test/Ungood/'
        elif self.data == 'liver':
            self.good = '/home/jinan/Datasets/graph_img/'
            self.ungood = '/home/jinan/Datasets/graph_img/'
            self.mask = '/home/jinan/Datasets/graph_img/'
        elif self.data == 'bras2021':
            self.good = '/home/jinan/Datasets/graph_img_brain/'
            self.ungood = '/home/jinan/Datasets/graph_img_brain/'
            self.mask = '/home/jinan/Datasets/graph_img_brain/'
        elif self.data == 'chest':
            self.good = '/home/jinan/Datasets/Medical-datasets/chest-rsna/Chest-RSNA/test/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/chest-rsna/Chest-RSNA/test/Ungood/'
        elif self.data == 'camelyon':
            self.good = '/home/jinan/Datasets/Medical-datasets/camelyon16_256/test/good/'
            self.ungood = '/home/jinan/Datasets/Medical-datasets/camelyon16_256/test/Ungood/'
        self.good_list = os.listdir(self.good)
        self.ungood_list = os.listdir(self.ungood)
                
    def load_image(self, path):
        image = Image.open(path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        if idx < len(self.ungood_list):
            img_path = self.ungood + self.ungood_list[idx]
            img = self.load_image(img_path)
            if self.data == 'OCT2017' or self.data == 'chest' or self.data == 'camelyon':
                mask = torch.zeros(1,256, 256)
            else:
                mask_path = self.mask + self.ungood_list[idx]
                mask_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                mask = mask_transform(Image.open(mask_path).convert('L'))
                mask[mask>=0.1] = 1
                mask[mask<0.1] = 0
            label = 1
        else:
            img_path = self.good + self.good_list[idx-len(self.ungood_list)]
            img = self.load_image(img_path)
            mask = torch.zeros(1,256, 256)
            label = 0
        #print(img_path)
        sample = {'image': img, 'mask': mask, 'label': label, 'path': str(img_path)}

        return sample

    def __len__(self):
       return len(self.good_list) + len(self.ungood_list)