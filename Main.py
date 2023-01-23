from modules.load_data import Generated_Train, Real_Test, Both_Train, get_stats
from modules.Train import train_net
from modules.get_transforms import get_light_transforms, get_strong_transforms
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--only_gen",action = 'store_true' , help= "if true, use only generated dataset")
    parser.add_argument("--augment", action = 'store_true', help= "if true, use classical augmentations")
    parser.add_argument("--mix_pct",type = float, default = 1, help= "amount of generated images to add. Between 0 and 1")
    parser.add_argument("--gen_path", type=str, help= "generated dataset path")
    parser.add_argument("--stl_train_path", type=str, help= "real train dataset path")
    parser.add_argument("--stl_test_path", type=str, help= "real test dataset path")

    opt = parser.parse_args()

    generated_train_path = opt.gen_path
    real_train_path = opt.stl_train_path
    real_test_path = opt.stl_test_path

    # define transforms
    # transforms_light is only for necesarry data processing (convert from int to float) 
    # transforms_strong includes data augmentation 
    transforms_light = get_light_transforms()
    transforms_strong = get_strong_transforms()

    if opt.augment:
        transforms_chosen = transforms_strong
    else:
        transforms_chosen = transforms_light

    # create test dataset
    # update transform normalizes the data
    test_dataset = Real_Test(real_test_path, transforms_light)
    
    #create train dataset
    if opt.only_gen:
        train_dataset = Generated_Train(generated_train_path, transforms_chosen)
    else:
        train_dataset = Both_Train(real_train_path, generated_train_path, opt.mix_pct, transforms_chosen)
        
    train_net(train_dataset,test_dataset)

if __name__ == "__main__":
    main()
