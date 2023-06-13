'''This is the repo which contains the original code to the WACV 2022 paper
"Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection"
by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
import torch
from train import train
from utils import load_datasets, make_dataloaders
from data import TrainDataset, ValidDataset, TestDataset, get_train_transforms, get_valid_transforms

train_dataset = TrainDataset(data='camelyon', transform=get_train_transforms())
valid_dataset = ValidDataset(data='camelyon', transform=get_valid_transforms())
test_dataset = TestDataset(data='camelyon', transform=get_valid_transforms())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=8)
# train_loader, test_loader = make_dataloaders(train_set, test_set)
train(train_loader, valid_loader, test_loader)
