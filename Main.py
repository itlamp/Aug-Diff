from load_data import Generated_Train, Real_Train, Real_Test, Both_Train, get_stats
from Train import train_net
import torch
import torchvision.transforms as transforms
import argparse

def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--only_gen",type = bool, help= "if true, use only generated dataset")
    parser.add_argument("--augment", type = bool, help= "if true, use classical augmentations")
    parser.add_argument("--mix_pct",type = bool, help= "amount of generated images to add. Between 0 and 1")
    parser.add_argument("--gen_path", type=str, help= "generated dataset path")
    parser.add_argument("--stl_train_path", type=str, help= "real train dataset path")
    parser.add_argument("--stl_test_path", type=str, help= "real test dataset path")

    opt = parser.parse_args()

    generated_train_path = opt.gen_path
    real_train_path = opt.stl_train_path
    real_test_path = opt.stl_test_path
    
    # define transforms
    # transforms_light is only for data conversion 
    transforms_light = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32)
    ])
    
    # transforms_strong includes data augmentation
    transforms_strong = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    ])

    if opt.augment:
        transforms_chosen = transforms_strong
    else:
        transforms_chosen = transforms_light


    test_dataloader = Real_Test(real_test_path, transforms_light)
    test_dataloader.update_transform()

    print(f"-----------------generated alone------------------")
    train_dataloader = Generated_Train(generated_train_path, transforms_light)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)


    print(f"-----------------generated augmentations------------------")
    train_dataloader = Generated_Train(generated_train_path, transforms_strong)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)


    print(f"-----------------both 0.1 alone------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 0.1, transforms_light)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)


    print(f"-----------------both 0.5 alone------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 0.5, transforms_light)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)

    print(f"-----------------both 1 alone------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 1, transforms_light)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)

    print(f"-----------------both 0.1 augmented------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 0.1, transforms_strong)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)

    print(f"-----------------both 0.5 augmented------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 0.5, transforms_strong)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)


    print(f"-----------------both 1 augmented------------------")
    train_dataloader = Both_Train(real_train_path, generated_train_path, 1, transforms_strong)
    train_dataloader.update_transform()
    train_net(train_dataloader,test_dataloader)

if __name__ == "__main__":
    main()
