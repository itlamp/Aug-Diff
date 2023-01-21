from load_data import Generated_Train, Real_Test, Both_Train, get_stats
from Train import train_net
from get_transforms import get_light_transforms, get_strong_transforms
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--only_gen",type = bool, default = False, help= "if true, use only generated dataset")
    parser.add_argument("--augment", type = bool, default = True, help= "if true, use classical augmentations")
    parser.add_argument("--mix_pct",type = float, default = 1, help= "amount of generated images to add. Between 0 and 1")
    parser.add_argument("--gen_path", type=str, help= "generated dataset path")
    parser.add_argument("--stl_train_path", type=str, help= "real train dataset path")
    parser.add_argument("--stl_test_path", type=str, help= "real test dataset path")

    opt = parser.parse_args()

    generated_train_path = opt.gen_path
    real_train_path = opt.stl_train_path
    real_test_path = opt.stl_test_path
    mix_pct = opt.mix_pct
    
    # define transforms
    # transforms_light is only for necesarry data processing (convert from int to float) 
    # transforms_strong includes data augmentation 
    transforms_light = get_light_transforms()
    transforms_strong = get_strong_transforms()

    if opt.augment:
        transforms_chosen = transforms_strong
    else:
        transforms_chosen = transforms_light

    # create test dataloader
    # update transform normalizes the data
    test_dataloader = Real_Test(real_test_path, transforms_light)
    
    #create train dataloader
    if opt.only_gen:
        train_dataloader = Generated_Train(generated_train_path, transforms_chosen)
    else:
        train_dataloader = Both_Train(real_train_path, generated_train_path, mix_pct, transforms_chosen)
        
    train_net(train_dataloader,test_dataloader)

    # print(f"-----------------generated alone------------------")
    # train_dataloader = Generated_Train(generated_train_path, transforms_light)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)


    # print(f"-----------------generated augmentations------------------")
    # train_dataloader = Generated_Train(generated_train_path, transforms_strong)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)


    # print(f"-----------------both 0.1 alone------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 0.1, transforms_light)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)


    # print(f"-----------------both 0.5 alone------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 0.5, transforms_light)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)

    # print(f"-----------------both 1 alone------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 1, transforms_light)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)

    # print(f"-----------------both 0.1 augmented------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 0.1, transforms_strong)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)

    # print(f"-----------------both 0.5 augmented------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 0.5, transforms_strong)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)


    # print(f"-----------------both 1 augmented------------------")
    # train_dataloader = Both_Train(real_train_path, generated_train_path, 1, transforms_strong)
    # train_dataloader.update_transform()
    # train_net(train_dataloader,test_dataloader)

if __name__ == "__main__":
    main()
