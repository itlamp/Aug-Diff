from load_data import Generated_Train, Real_Train, Real_Test, Both_Train, get_stats
from Train import train_net
import torch
import torchvision.transforms as transforms

transforms_light = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32)
])

transforms_strong = transforms.Compose([
transforms.ConvertImageDtype(torch.float32),
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomApply([
transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
], p=0.8),
transforms.RandomGrayscale(p=0.2),
])

real_test_path = '/databases/itayl/stl_data/test_set'
real_train_path = '/databases/itayl/stl_data/train_set'
generated_train_path = '/databases/itayl/generated_dataset'

test_dataloader = Real_Test(real_test_path, transforms_light)
test_dataloader.update_transform()


# print(f"-----------------real alone------------------")
# train_dataloader = Real_Train(real_train_path, transforms_light)
# train_dataloader.update_transform()
# train_net(train_dataloader,test_dataloader)


# print(f"-----------------real augmentations------------------")
# train_dataloader = Real_Train(real_train_path, transforms_strong)
# train_dataloader.update_transform()
# train_net(train_dataloader,test_dataloader)

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