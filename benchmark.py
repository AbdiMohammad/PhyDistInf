import json
import sys
import os
import random
import numpy as np
import pathlib
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

from PhyDistInf import CodebookTrainData, cifar_resnet_loader_generator, create_resnet_with_codebook, get_codebook_params_and_ids, evaluate_codebook_model
from PhyDistInf import ConstantCoefficient, LinearCoefficient
from codebook_output import CodebookOutput

from bottlefit import create_resnet_with_bottlefit

def benchmark_model(model, dataloader, method, latent_layer_name=None, save_dir=None, save_inference_time=True, save_latent=False):
    device = next(model.parameters()).device

    if args.method == "SC":
        exec(f"model.{latent_layer_name} = SplitLayer(model.{latent_layer_name})")
    elif "BF" in args.method:
        exec(f"model.{latent_layer_name}.bn1 = SplitLayer(model.{latent_layer_name}.bn1)")

    if save_latent:
        save_dir_latent = os.path.join(save_dir, "transmit_data")
        pathlib.Path(save_dir_latent).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i, (xs, _) in enumerate(dataloader):
                if method == "EC":
                    torch.save(xs, os.path.join(save_dir_latent, f"{i}.pth"))
                    continue
                xs = xs.to(device)
                _ = model(xs)
                latent = torch.load(".temp/latent.pth")
                os.remove(".temp/latent.pth")
                torch.save(latent, os.path.join(save_dir_latent, f"{i}.pth"))
                # print(latent.shape)

    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    head_inference_time, tail_inference_time, total_inference_time = [], [], []

    # GPU Warm-up
    with torch.no_grad():
        for xs, _ in dataloader:
            xs = xs.to(device)
            _ = model(xs)

    with torch.no_grad():
        for repretitions in range(10):
            for xs, _ in dataloader:
                xs = xs.to(device)
                start_event.record()
                out = model(xs)
                end_event.record()
                torch.cuda.synchronize()
                if method == "SC":
                    split_event = eval(f"model.{latent_layer_name}.split_event")
                    head_inference_time.append(start_event.elapsed_time(split_event))
                    tail_inference_time.append(split_event.elapsed_time(end_event))
                elif "BF" in method:
                    split_event = eval(f"model.{latent_layer_name}.bn1.split_event")
                    head_inference_time.append(start_event.elapsed_time(split_event))
                    tail_inference_time.append(split_event.elapsed_time(end_event))
                elif method == "PhyDistInf":
                    transmit_event = out.codebook_outputs[0][2].transmit_event
                    receive_event = out.codebook_outputs[0][2].receive_event
                    head_inference_time.append(start_event.elapsed_time(transmit_event))
                    tail_inference_time.append(receive_event.elapsed_time(end_event))
                
                total_inference_time.append(start_event.elapsed_time(end_event))

    if args.method == "SC":
        exec(f"model.{latent_layer_name} = model.{latent_layer_name}.split_layer")
    elif "BF" in args.method:
        exec(f"model.{latent_layer_name}.bn1 = model.{latent_layer_name}.bn1.split_layer")
    
    head_inference_time = torch.Tensor(head_inference_time)
    tail_inference_time = torch.Tensor(tail_inference_time)
    total_inference_time = torch.Tensor(total_inference_time)

    if save_inference_time:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        inference_time_dict = {
            'head': head_inference_time,
            'tail': tail_inference_time,
            'total': total_inference_time
        }
        torch.save(inference_time_dict, os.path.join(save_dir, "inference_time.pth"))

    return head_inference_time, tail_inference_time, total_inference_time

def profile_model(model, dataloader):
    device = next(model.parameters()).device

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=2, repeat=2),
                     on_trace_ready=trace_handler) as p:
            for i, (xs, _) in enumerate(dataloader):
                xs = xs.to(device)
                model(xs)
                p.step()
                if i == 19:
                    break

class SplitLayer(nn.Module):

    def __init__(self, split_layer):
        super().__init__()
        self.split_layer = split_layer
        self.split_event = torch.cuda.Event(enable_timing=True)
    
    def forward(self, x):
        res = self.split_layer(x)
        self.split_event.record()
        # torch.save(res[:, torch.randint(high=res.shape[1], size=(self.layer_width,)), :, :], ".temp/latent.pth")
        # torch.save(res, ".temp/latent.pth")
        return res

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, default="PhyDistInf", choices=["EC", "SC", "BF-2", "BF-1", "PhyDistInf"])

    args = parser.parse_args()

    set_seed(0)

    with open(args.config) as config_file:
        json_data = json.load(config_file)

    device = torch.device('cuda')

    dataset = json_data["dataset"]
    BATCH_SIZE = 1 # json_data["batch_size"]
    model_name = json_data["model"]

    reference_pretrained_weights_path = os.path.join(json_data['reference_pretrained_weights_dir'], dataset, model_name) + ".pth"

    if dataset == 'cifar10':
        normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize
        ])
        cifar10_test = torchvision.datasets.CIFAR10('dataset/cifar10', train=False, download=True, transform=transform)
        cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=16)

        valid_dl = cifar10_test_dataloader

        model = cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path)()
    elif dataset == 'cifar100':
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        cifar100_test = torchvision.datasets.CIFAR100('dataset/cifar100', train=False, download=True, transform=transform)
        cifar100_test_dataloader = DataLoader(cifar100_test, batch_size=BATCH_SIZE, num_workers=16)

        valid_dl = cifar100_test_dataloader

        model = cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path, num_classes=100)()

    if args.method == "PhyDistInf":
        PSNR = json_data['PSNR']
        pretrained_weights_path = os.path.join(json_data['output_dir'], "model.pth")
        pretrained_weights = torch.load(pretrained_weights_path, map_location=device)

        codebook_sizes = [pretrained_weights[key].shape[0] for key in pretrained_weights if 'embedding' in key]

        codebook_training_data = []

        for codebook_training_data_json, codebook_size in zip(json_data['codebooks'], codebook_sizes):
            codebook_training_data.append(
                CodebookTrainData(
                    codebook_training_data_json['layer'],
                    hidden_dim=codebook_training_data_json['hidden_dim'],
                    codebook_size=codebook_size,
                    beta=eval(codebook_training_data_json['beta']),
                    prune_epochs=codebook_training_data_json['prune_epochs'],
                    prune_values=[codebook_training_data_json['prune_value'] / len(codebook_training_data_json['prune_epochs'])] * len(codebook_training_data_json['prune_epochs']),
                    prune_values_type=codebook_training_data_json['prune_value_type']
                )
            )

        codebook_lr = json_data['codebook_lr']
        non_codebook_lr = json_data['non_codebook_lr']

        if dataset == 'cifar10':
            model = create_resnet_with_codebook(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path), PSNR, codebook_training_data)
        elif dataset == 'cifar100':
            model = create_resnet_with_codebook(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path, num_classes=100), PSNR, codebook_training_data)

        model.load_state_dict(pretrained_weights)

        all_params = list(model.parameters())

        codebook_params, codebook_ids = get_codebook_params_and_ids(model)
        non_codebook_params = [p for p in all_params if id(p) not in codebook_ids]

        optimizer = torch.optim.Adam([
            {'params': non_codebook_params, 'lr': non_codebook_lr},
            {'params': codebook_params, 'lr': codebook_lr},
        ])
    elif "BF" in args.method:
        model = create_resnet_with_bottlefit(model, json_data['codebooks'][0]['layer'], int(args.method.split('-')[1]))
        model.load_state_dict(torch.load(os.path.join(json_data['output_dir'], f"{model_name}_BF-{args.method.split('-')[1]}.pth"))['state_dict'])

    head_inference_time, tail_inference_time, total_inference_time = benchmark_model(model, valid_dl, args.method, latent_layer_name=json_data['codebooks'][0]['layer'], save_dir=os.path.join(json_data['output_dir'], f"{args.method}_{model_name}"))

    print(f"head inference time averaged over {len(head_inference_time)}: {head_inference_time.mean()}")
    print(f"tail inference time averaged over {len(tail_inference_time)}: {tail_inference_time.mean()}")
    print(f"total inference time averaged over {len(total_inference_time)}: {total_inference_time.mean()}")

    # profile_model(resnet_with_codebook, valid_dl)

