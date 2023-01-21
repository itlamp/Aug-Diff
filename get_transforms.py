import torchvision.transforms as transforms
import torch

# transforms_light includes data augmentation necessary for proper function of the CNN. 
# this is only converstion to float
def get_light_transforms(): 
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32)
    ])

# transforms_strong includes data augmentation 
def get_strong_transforms(): 
    return transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    ])
