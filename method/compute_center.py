import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.res12 import Res12
from models.resnet import resnet50
from models.swin_transformer import swin_tiny
import torch.utils.data
from utils.utils import transform_val_cifar, cluster
from utils.utils import transform_val_224_cifar


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backbone == 'resnet12':
        model = Res12(avg_pool=True, drop_block='ImageNet' in args.dataset).to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k[8:]: v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    elif args.backbone == 'resnet50':
        model = resnet50().to(device)
        checkpoint = torch.load('./checkpoints/resnet50-0676ba61.pth')
    elif args.backbone == 'swin':
        model = swin_tiny().to(device)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.model_path)['params']
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    data = {}
    batch_size = 128
    shuffle = True

    # train
    if args.dataset == 'FC100':
        train_set = ImageFolder(r'C:\workspace\code\PythonProject\SemFew\dataset\FC100\FC1001\train',
                                transform=transform_val_cifar if args.backbone == 'resnet12' or args.backbone == 'resnet50' else transform_val_224_cifar)
    elif args.dataset == 'CIFAR-FS':
        train_set = ImageFolder(r'C:\workspace\code\PythonProject\SemFew\dataset\CIFAR-FS\cifar-fs\train',
                                transform=transform_val_cifar if args.backbone == 'resnet12' or args.backbone == 'resnet50' else transform_val_224_cifar)
    else:
        raise ValueError('Non-supported Dataset.')

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                              pin_memory=True)
    idx_to_class = train_set.class_to_idx
    idx_to_class = {k: v for v, k in idx_to_class.items()}
    for x, labels in tqdm(train_loader):
        labels = [idx_to_class[l.item()] for l in labels]
        with torch.no_grad():
            x = model(x.to(device))
        for i, l in enumerate(labels):
            if l in data:
                data[l].append(x[i].detach().cpu().numpy())
            else:
                data[l] = [x[i].detach().cpu().numpy()]

    print('Finished train')

    center_mean = {}
    for k, v in data.items():
        center_mean[k] = np.array(v).mean(0)

    center_cluster = cluster(data, len(data), 600)

    torch.save({
        'mean': center_mean,
        'cluster': center_cluster,
        'center': center_mean
    }, './center/center_{}_{}.pth'.format(args.dataset, args.backbone))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FC100',
                        choices=['FC100', 'CIFAR-FS'])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--center', default='mean',
                        choices=['mean', 'cluster'])
    parser.add_argument('--backbone', default='resnet12',
                        choices=['resnet12', 'swin', 'resnet50'])
    args = parser.parse_args()
    print(vars(args))
    if args.backbone == 'resnet12':
        args.model_path = './checkpoints/ResNet-{}.pth'.format(args.dataset)
    elif args.backbone == 'swin':
        args.model_path = './checkpoints/Swin-Tiny-{}.pth'.format(args.dataset)
    main()
