import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
import os
import pandas as pd
from torchvision.io import read_image
import glob
import numpy as np
 
from pathlib import Path
class Generated_Train(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __len__(self):
        return 500*10

    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        index = int(np.floor(idx/500))
        label = label_list[index]
        img_path = glob.glob(self.img_dir + '/' + label + '/samples/*.png')[idx%500]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, index
    
    
class Real_Train(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __len__(self):
        return 500*10

    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        index = int(np.floor(idx/500))
        label = label_list[index]
        img_path = glob.glob(self.img_dir + '/' + label + '/*.png')[idx%500]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, index
    
    
class Real_Test(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __len__(self):
        return 800*10

    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        index = int(np.floor(idx/800))
        label = label_list[index]
        img_path = glob.glob(self.img_dir + '/' + label + '/*.png')[idx%800]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, index
    
    
class Both_Train(Dataset):
    def __init__(self, real_dir, gen_dir, mix_pct, transform=None):
        self.real_dir = real_dir
        self.gen_dir = gen_dir
        self.transform = transform
        self.class_size = int(500+500*mix_pct)


    def __len__(self):
        return (self.class_size)*10

    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        label_num = int(np.floor(idx/self.class_size))
        label = label_list[label_num]
        if idx%self.class_size < 500:
            path = self.real_dir + '/' + label + '/*.png'
            img_path = glob.glob(path)[idx%self.class_size]
        else:
            path = self.gen_dir + '/' + label + '/samples/*.png'
            img_path = glob.glob(path)[idx%self.class_size-500]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label_num

def get_stats(image_data):
    image_data_loader = DataLoader(
    image_data, 
    # batch size is whole datset
    batch_size=len(image_data), 
    shuffle=False, 
    num_workers=0)

    def mean_std(loader):
        images, lebels = next(iter(loader))
        # shape of images = [b,c,w,h]
        images = images.to(torch.float32)
        mean, std = images.mean([0,2,3]), images.std([0,2,3])
        return mean, std

    mean, std = mean_std(image_data_loader)
    # print("mean and std: \n", mean, std)
    return mean, std