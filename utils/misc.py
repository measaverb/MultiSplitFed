import json

import numpy as np
import torch
from torchvision import transforms


def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parameters(net):
    grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters()],
        }
    ]
    return grouped_parameters


def get_optimizer(config, net):
    optimizer = torch.optim.Adam(
        get_parameters(net),
        lr=config["networks"]["lr"],
    )
    return optimizer


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Pad(3),
            transforms.RandomRotation(10),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Pad(3),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transforms, test_transforms
