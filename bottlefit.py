import sys
import json
import pathlib
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from PhyDistInf import cifar_resnet_loader_generator
from PhyDistInf import model_device

def train_model(model, n_epochs, train_dl, valid_dl, optimizer):
    device = model_device(model)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(3.0 / 5 * n_epochs), round(4.0 / 5 * n_epochs)], gamma=0.1)

    best_acc = -1
    for epoch in range(n_epochs):
        for xs, labels in tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}'):
            xs = xs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model_output = model(xs)
            loss = F.cross_entropy(model_output, labels)
            loss.backward()
            optimizer.step()

        n_correct, n_all = 0, 0
        with torch.no_grad():
            for xs, labels in tqdm(valid_dl, desc=f'Epoch {epoch}/{n_epochs}'):
                xs = xs.to(device)
                labels = labels.to(device)
                model_output = model(xs)
                n_correct += (model_output.argmax(dim=-1) == labels).float().sum().item()
                n_all += len(labels)
        
        acc = n_correct / n_all
        if acc > best_acc:
            best_acc = acc
            print(f"best acc: {best_acc}")

        scheduler.step()
    
    return best_acc

def create_resnet_with_bottlefit(pretrained_resnet, bottle_layer, bottle_width):
    device = torch.device('cuda')

    curr_block = eval(f"pretrained_resnet.{bottle_layer}")
    curr_block.conv1 = nn.Conv2d(16, bottle_width, kernel_size=3, stride=1, padding=1, bias=False)
    curr_block.bn1 = nn.BatchNorm2d(bottle_width)
    curr_block.conv2 = nn.Conv2d(bottle_width, 16, kernel_size=3, stride=1, padding=1, bias=False)
    curr_block.bn2 = nn.BatchNorm2d(16)

    return pretrained_resnet.to(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bottle_width", type=int, default=1)

    args = parser.parse_args()

    json_args_file_path = args.config
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    dataset = json_data["dataset"]
    BATCH_SIZE = json_data["batch_size"]
    model_name = json_data["model"]

    device = torch.device('cuda')

    if dataset == 'cifar10':
        normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize
        ])
        cifar10_train = torchvision.datasets.CIFAR10('dataset/cifar10', train=True, download=True, transform=transform)
        cifar10_test = torchvision.datasets.CIFAR10('dataset/cifar10', train=False, download=True, transform=transform)
        cifar10_train_dataloader = DataLoader(cifar10_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
        cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=16)

        train_dl = cifar10_train_dataloader
        valid_dl = cifar10_test_dataloader
    
    if dataset == 'cifar100':
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        cifar100_train = torchvision.datasets.CIFAR100('dataset/cifar100', train=True, download=True, transform=transform)
        cifar100_test = torchvision.datasets.CIFAR100('dataset/cifar100', train=False, download=True, transform=transform)
        cifar100_train_dataloader = DataLoader(cifar100_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
        cifar100_test_dataloader = DataLoader(cifar100_test, batch_size=BATCH_SIZE, num_workers=16)

        train_dl = cifar100_train_dataloader
        valid_dl = cifar100_test_dataloader

    save_weights_after_train = json_data['save_weights_after_train']
    n_epochs = 100 # json_data['n_epochs']

    output_dir = json_data['output_dir']
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    reference_pretrained_weights_path = os.path.join(json_data['reference_pretrained_weights_dir'], dataset, model_name) + ".pth"

    if dataset == 'cifar10':
        model = create_resnet_with_bottlefit(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path)(), json_data['codebooks'][0]['layer'], args.bottle_width)
    elif dataset == 'cifar100':
        model = create_resnet_with_bottlefit(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path, num_classes=100)(), json_data['codebooks'][0]['layer'], args.bottle_width)

    optimizer = torch.optim.Adam(model.parameters())

    best_acc = train_model(model, n_epochs, train_dl, valid_dl, optimizer)
    
    save_data = {
        'state_dict': model.state_dict(),
        'best_acc': best_acc
    }
    if save_weights_after_train:
        torch.save(save_data, f'{output_dir}/{model_name}_BF-{args.bottle_width}.pth')