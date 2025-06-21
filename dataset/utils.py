import os
import torch
import torchvision


# ----------------------------- transform -------------------------------- #
# imagenet transform
transform_imagenet_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_imagenet_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ----------------------------------------------------------------------- #


def build_dataloader_imagenet(data_dir, conf_num=5000, temp_num=5000, batch_size=512, num_workers=8):
    validir = os.path.join(data_dir, 'imagenet/images/val')
    testset = torchvision.datasets.ImageFolder(root=validir, transform=transform_imagenet_test)

    dataset_length = len(testset)
    cal_num = conf_num + temp_num
    calibset, testset = torch.utils.data.random_split(testset, [cal_num, dataset_length - cal_num])
    conf_calibset, calib_calibset = torch.utils.data.random_split(calibset, [conf_num, cal_num - conf_num])

    calib_calibloader = torch.utils.data.DataLoader(dataset=calib_calibset, batch_size=batch_size,
                                                    num_workers=num_workers)
    conf_calibloader = torch.utils.data.DataLoader(dataset=conf_calibset, batch_size=batch_size,
                                                   num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)

    return calib_calibloader, conf_calibloader, testloader
