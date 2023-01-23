import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import glob
import numpy as np
 
class Generated_Train(Dataset):
    """"Torch Dataset for generated images only"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.update_transform()

    # After dataset is created, normalizes the dataset according to emperical distribution
    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __len__(self):
        return 500*10

    # Our __getitem__ assumes that all of the images are in a file with the corresponding name. 
    # idx 0-499 corresponds to airplane, 500-999 to bird, etc.
    # returns img and label
    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        label = int(np.floor(idx/500))
        label_name = label_list[label]
        img_path = glob.glob(self.img_dir + '/' + label_name + '/*.png')[idx%500]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label


class Real_Test(Dataset):
    """"Torch Dataloader for test images only"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.update_transform()

    # After dataset is created, normalizes the dataset according to emperical distribution
    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    def __len__(self):
        return 800*10

    # Our __getitem__ assumes that all of the images are in a file with the corresponding name. 
    # idx 0-799 corresponds to airplane, 800-1599 to bird, etc.
    # returns img and label
    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        label = int(np.floor(idx/800))
        label_name = label_list[label]
        img_path = glob.glob(self.img_dir + '/' + label_name + '/*.png')[idx%800]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label
    
class Both_Train(Dataset):
    """"Torch Dataloader for generated and original train images"""

    ## receives two paths and mix percentage (unlike datasets above)
    def __init__(self, real_dir, gen_dir, mix_pct, transform=None):
        self.real_dir = real_dir
        self.gen_dir = gen_dir
        self.transform = transform
        self.class_size = int(500+500*mix_pct)
        self.update_transform()

    def __len__(self):
        return (self.class_size)*10

    def update_transform(self):
        mean, std = get_stats(self)
        self.transforms = transforms.Compose([self.transform,transforms.Normalize(mean,std)])
    
    
    # Our __getitem__ assumes that all of the images are in a file with the corresponding name. 
    # idx 0 to 499 corresponds to real airplane, idx 500 to 500*mix_pct correspond to generated airplane
    # idx 500*mix_pct to 500*mix_pct + 500 correspond to real bird, the next 500*mix_pct indices correspond to generated bird
    # etc.
    # returns img and label
    def __getitem__(self, idx):
        label_list = ['airplane','bird','car','cat','deer','dog','horse','monkey','ship','truck']
        label = int(np.floor(idx/self.class_size))
        label_name = label_list[label]
        if idx%self.class_size < 500:
            path = self.real_dir + '/' + label_name + '/*.png'
            img_path = glob.glob(path)[idx%self.class_size]
        else:
            path = self.gen_dir + '/' + label_name + '/*.png'
            img_path = glob.glob(path)[idx%self.class_size-500]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

## receives a dataset and returns its mean and std
def get_stats(image_data):
    #create dataloader
    image_data_loader = DataLoader(
    image_data, 
    # batch size is whole datset
    batch_size=len(image_data), 
    shuffle=False, 
    num_workers=0)

    ## receives dataloader, iterates over all images and calculates mean and std
    def mean_std(loader):
        images, _ = next(iter(loader))
        # shape of images = [b,c,w,h]
        images = images.to(torch.float32)
        mean, std = images.mean([0,2,3]), images.std([0,2,3]) # don't take mean over channels
        return mean, std
    
    ## calculate meand and std over whole dataset
    mean, std = mean_std(image_data_loader)
    return mean, std