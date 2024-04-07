#%%
import torch
import numpy as np
import random
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pickle
import torchvision.datasets
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pathlib
import math
import torch.distributions
import sklearn
import sklearn.cluster
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Callable
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# from torchvision.models import resnet18, resnet152
# import torch_pruning
from torch.autograd import Variable
from IPython.display import display
from ipywidgets import interactive
import json
import sys
import os
from codebook_output import CodebookOutput
import cifar_resnet
import imagenet_resnet

from codebook import Codebook
from utils import op_counter
from utils import ptflops

def LinearCoefficient(start_beta, end_beta):
    def res(epoch, n_epochs):
        return start_beta + (end_beta - start_beta) * epoch / n_epochs
    return res

def ConstantCoefficient(beta):
    def res(epoch, n_epochs):
        return beta
    return res

def categorical_entropy(dist):
    return torch.distributions.Categorical(dist).entropy().mean(dim=-1)

class StopCompute(nn.Module):

    def __init__(self, inner):
        super().__init__()

        self.inner = inner
    
    def forward(self, x):
        res = self.inner(x)
        raise Exception(res)
    
def cifar_resnet_loader_generator(model_name, pretrained_weights_path=None, num_classes=10):
    def load_pretrained_resnet():
        device = torch.device('cuda')
        # rn = cifar_resnet20()
        rn = eval(f'cifar_resnet.cifar_{model_name}')(num_classes=num_classes)
        rn = rn.to(device)
        def remove_module_prefix(d):
            res = dict()
            for key in d.keys():
                res[key[len('module.'):]] = d[key]
            return res
        if pretrained_weights_path is not None:
            model_dict = torch.load(pretrained_weights_path)
            if num_classes == 10:
                rn.load_state_dict(remove_module_prefix(model_dict['state_dict']))
            elif num_classes == 100:
                rn.load_state_dict(model_dict)
        return rn
    return load_pretrained_resnet

def imagenet_resnet_loader_generator(model_name):
    def load_pretrained_imagenet():
        device = torch.device('cuda')
        rn = eval('imagenet_resnet.imagenet_' + model_name)()
        print("loaded model modules size:", len(list(rn.modules())))
        rn = rn.to(device)
        return rn
    return load_pretrained_imagenet

def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    n_all = 0
    n_correct = 0
    for xs, labels in tqdm(dataloader):

        xs = xs.to(device)
        labels = labels.to(device)
        out = model(xs)
        n_correct += (out.argmax(dim=-1) == labels).sum().item()
        n_all += len(xs)
    return n_correct / n_all

def evaluate_codebook_model(model, dataloader, codebook_index=-1):
    device = next(model.parameters()).device
    n_all = 0
    n_correct = 0
    with torch.no_grad():
        for xs, labels in tqdm(dataloader):

            xs = xs.to(device)
            labels = labels.to(device)
            out = model(xs)
            if type(out) == CodebookOutput:
                if codebook_index == -1:
                    out = out.original_tensor
                else:
                    out = out.codebook_outputs[codebook_index][0]
            n_correct += (out.argmax(dim=-1) == labels).sum().item()
            n_all += len(xs)
    return n_correct / n_all

def model_device(model):
    return next(model.parameters()).device

def resnet_vib_loss(model_output: CodebookOutput, labels, epoch, n_epochs):
    if type(model_output) != CodebookOutput:
        dummy = CodebookOutput(model_output, [])
        return resnet_vib_loss(dummy, labels)

    original_model_outputs = model_output.original_tensor
    metrics = dict()
    original_model_loss = F.cross_entropy(original_model_outputs, labels)
    original_model_n_correct = (original_model_outputs.argmax(dim=-1) == labels).float().sum().item()
    loss = original_model_loss
    metrics["original"] = original_model_n_correct

    for codebook_output, dist, codebook in model_output.codebook_outputs:
        distortion_loss = F.cross_entropy(codebook_output, labels)
        codebook_n_correct = (codebook_output.argmax(dim=-1) == labels).float().sum().item()
        codebook_entropy = categorical_entropy(dist).mean().item()
        codebook_loss = distortion_loss + codebook.beta(epoch, n_epochs) * codebook_entropy
        loss += codebook_loss
        metrics["codebook at " + codebook.train_data.layer_name] = codebook_n_correct
    return loss, metrics

def train_model(n_epochs, model, loss_fn, train_dataloader, valid_dataloader, optimizer=None, codebook_training_datas=[]):
    device = model_device(model)
    train_losses = []
    train_metrics = defaultdict(list)
    valid_losses = []
    valid_metrics = defaultdict(list)
    output_names = get_output_names(model, train_dataloader)
    best_acc = -1
    fine_tune_epoch = np.array([prune_data.prune_epochs for prune_data in codebook_training_datas]).max()

    writer = SummaryWriter()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(3.0 / 5 * n_epochs), round(4.0 / 5 * n_epochs)], gamma=0.1)

    for epoch in range(n_epochs):
        for i, codebook_training_data in enumerate(codebook_training_datas):
            try:
                prune_index = codebook_training_data.prune_epochs.index(epoch)
            except ValueError:
                prune_index = -1

            if prune_index != -1:
                print('_' * 80)
                print('Pruning codebook {} at epoch {}, removing {} codewords ({})'.format(codebook_training_data.layer_name, epoch,\
                                                                                          codebook_training_data.prune_values[prune_index], codebook_training_data.prune_values_type))
                # print('Valid accuracy before pruning: ', evaluate_codebook_model(model, valid_dataloader, i))
                prune_and_replace_codebook(model, train_dataloader, i, codebook_training_data.layer_name,\
                                           codebook_training_data.prune_values[prune_index], codebook_training_data.prune_values_type)
                # print('Valid accuracy after pruning: ', evaluate_codebook_model(model, valid_dataloader, i))
                print('_' * 80)

        for output_name in output_names:
            train_metrics[f'{output_name} n_correct'] = 0
        train_n_all = 0
        for xs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch}/{n_epochs}'):

            xs = xs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            model_output = model(xs)
            loss, metrics = loss_fn(model_output, labels, epoch, n_epochs)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            writer.add_scalar('train loss', loss.item(), epoch)

            train_n_all += len(labels)
            for output_name, metric in metrics.items():
                train_metrics[output_name + ' acc_batch'].append(metric / len(labels))
                train_metrics[output_name + ' n_correct'] += metric
                writer.add_scalar(output_name + '/train', metric, epoch)
        
        for output_name, metric in metrics.items():
            train_metrics[output_name + ' acc'].append(train_metrics[output_name + ' n_correct'] / train_n_all)
            del train_metrics[output_name + ' n_correct']

        for output_name in output_names:
            valid_metrics[f'{output_name} n_correct'] = 0
        valid_n_all = 0
        with torch.no_grad():
            for xs, labels in tqdm(valid_dataloader, desc=f'Epoch {epoch}/{n_epochs}'):
                xs = xs.to(device)
                labels = labels.to(device)
                model_output = model(xs)
                loss, metrics = loss_fn(model_output, labels, epoch, n_epochs)
                valid_losses.append(loss.item())
                writer.add_scalar('valid loss', loss.item(), epoch)

                valid_n_all += len(labels)
                for output_name, metric in metrics.items():
                    valid_metrics[output_name + ' acc_batch'].append(metric / len(labels))
                    valid_metrics[output_name + ' n_correct'] += metric
                    writer.add_scalar(output_name + '/valid', metric, epoch)

        for output_name, metric in metrics.items():
            valid_metrics[output_name + ' acc'].append(valid_metrics[output_name + ' n_correct'] / valid_n_all)
            if 'codebook' in output_name:
                print(f"curr_val_acc: {valid_metrics[output_name + ' acc'][-1]}")
            del valid_metrics[output_name + ' n_correct']
            if epoch > fine_tune_epoch and 'codebook' in output_name and valid_metrics[output_name + ' acc'][-1] > best_acc:
                best_acc = valid_metrics[output_name + ' acc'][-1]
                # print(f"New best_acc: {best_acc}")

        scheduler.step()
        
    writer.close()
    return train_metrics, train_losses, valid_metrics, valid_losses, best_acc

@dataclass
class CodebookTrainData:
    layer_name: str
    hidden_dim: int
    codebook_size: int
    beta: Callable[[int, int], float]
    prune_epochs: List[int]
    prune_values: List[float]
    prune_values_type: str

def create_resnet_with_codebook(loader_fn, PSNR, codebooks: List[CodebookTrainData], dataloader=None):
    unmodified_resnet = loader_fn()
    pretrained_resnet = loader_fn()

    for codebook in codebooks:
        target_module = eval('pretrained_resnet.' + codebook.layer_name)

        # n_channels = get_layer_output_shape(unmodified_resnet, dataloader, codebook.layer_name)[1]
        
        new_module = nn.Sequential(
            target_module,
            #FIXME: Determine input dimension of codebook either based on config file or based on last layer's number of channels
            Codebook(codebook.hidden_dim, codebook.codebook_size, codebook.beta, codebook, PSNR=PSNR).to(model_device(pretrained_resnet))
        )
        exec('pretrained_resnet.' + codebook.layer_name + '= new_module')
        if dataloader is not None:
            # Initialize the codebook
            weights = get_initial_weights(unmodified_resnet, dataloader, codebook.layer_name, codebook.hidden_dim, codebook.codebook_size)
            exec('pretrained_resnet.' + codebook.layer_name + '[-1].embedding.data = torch.Tensor(weights).to(model_device(pretrained_resnet))')
    return pretrained_resnet

def get_initial_weights(model, dataloader, layer, codebook_dim, codebook_size):
    embeddings_list = []
    exec('model.' + layer + '= StopCompute(model.' + layer + ')')

    for xs, labels in tqdm(dataloader):
        xs = xs.to(model_device(model))
        labels = labels.to(model_device(model))
        try:
            _ = model(xs)
        except Exception as e:
            embeddings_list.append(e.args[0].detach())

    embeddings = torch.cat(embeddings_list).cpu().numpy().reshape(-1, codebook_dim)
    k_means = sklearn.cluster.MiniBatchKMeans(n_clusters=codebook_size, n_init='auto')
    k_means.fit(embeddings)

    exec('model.' + layer + '= model.' + layer + '.inner')
    return k_means.cluster_centers_

def get_layer_output_shape(model, dataloader, layer):
    exec('model.' + layer + '= StopCompute(model.' + layer + ')')

    try:
        _ = model(next(iter(dataloader))[0].to(model_device(model)))
    except Exception as e:
        if type(e.args[0]) == CodebookOutput:
            return e.args[0].original_tensor.shape
        else:
            return e.args[0].detach().shape
    finally:
        exec('model.' + layer + '= model.' + layer + '.inner')

def get_codebook_params_and_ids(model):

    ids = []
    params = []

    for module in model.modules():
        if type(module) == Codebook:
            for param in module.parameters():
                ids.append(id(param))
                params.append(param)
    return params, ids

def get_codebook_usage_data(model, dataloader, codebook_index):
    with torch.no_grad():
        sample_output = model(next(iter(dataloader))[0].to(model_device(model)))
        codebook_dim = sample_output.codebook_outputs[codebook_index][1].shape[1]
        counts = torch.zeros(codebook_dim, dtype=torch.int64).to(model_device(model))

        for xs, _ in tqdm(dataloader):
            xs = xs.to(model_device(model))
            output = model(xs)
            dist = output.codebook_outputs[codebook_index][1]
            batch_indices, batch_counts = dist.argmax(dim=1).unique(return_counts=True)
            counts[batch_indices] += batch_counts
        return counts
    
def prune_codebook(model, dataloader, codebook_index, value_to_prune, value_type):
    sample_output = model(next(iter(dataloader))[0].to(model_device(model)))
    codebook = sample_output.codebook_outputs[codebook_index][2]

    counts = get_codebook_usage_data(model, dataloader, codebook_index)
    if value_type == 'number':
        unpruned_indices = counts.sort()[1][round(value_to_prune):]
    elif value_type == 'percentage':
        unpruned_indices = counts.sort()[1][round(value_to_prune/100.0*len(counts)):]
        # unpruned_indices = (counts > (value_to_prune / 100.0 * counts.max())).nonzero(as_tuple=True)[0]

    new_codebook = Codebook(codebook.latent_dim, codebook.n_embeddings - (len(counts) - len(unpruned_indices)), codebook.beta, codebook.train_data).to(model_device(model))
    new_codebook.embedding.data = codebook.embedding.data[unpruned_indices]
    print(f'{len(counts) - counts.count_nonzero().item()} unused codewords found')
    print(f'{len(counts) - len(unpruned_indices)} codewords removed')
    # codebook = model.layer1[0][1]
    # codebook.embedding.data[indices_to_prune] = 0
    return new_codebook

def prune_and_replace_codebook(model, dataloader, index, layer_name, value_to_prune, value_type='number'):
    codebook_layer_name = layer_name + '[-1]'
    exec('model.' + codebook_layer_name + '= prune_codebook(model, dataloader, index, value_to_prune, value_type)')

def get_output_names(model, data_loader):
    device = model_device(model)
    sample_output = model(next(iter(data_loader))[0].to(device))
    output_names = ["original"]
    if type(sample_output) == CodebookOutput:
        for _, _, codebook in sample_output.codebook_outputs:
            output_names.append(f"codebook at {codebook.train_data.layer_name}")
    return output_names

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

#%%
# pretrained_resnet = load_pretrained_resnet()

if __name__ == '__main__':

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    dataset = json_data["dataset"]
    BATCH_SIZE = json_data["batch_size"]
    model_name = json_data["model"]

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

    if dataset == 'imagenet':
        # todo: load actual imagenet
        imagenet_train = TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
        imagenet_test = TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
        imagenet_train_dataloader = DataLoader(imagenet_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16) 
        imagenet_test_dataloader = DataLoader(imagenet_test, batch_size=BATCH_SIZE, num_workers=16)

        train_dl = imagenet_train_dataloader
        valid_dl = imagenet_test_dataloader


    
    PSNR = json_data['PSNR']
    pretrained_weights_path = json_data['pretrained_weights_path']
    save_weights_after_train = json_data['save_weights_after_train']
    n_epochs = json_data['n_epochs']
    codebook_lr = json_data['codebook_lr']
    non_codebook_lr = json_data['non_codebook_lr']

    output_dir = json_data['output_dir']
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    reference_pretrained_weights_path = os.path.join(json_data['reference_pretrained_weights_dir'], dataset, model_name) + ".pth"

    codebook_training_data = []

    for codebook_training_data_json in json_data['codebooks']:
        codebook_training_data.append(
            CodebookTrainData(
                codebook_training_data_json['layer'],
                hidden_dim=codebook_training_data_json['hidden_dim'],
                codebook_size=codebook_training_data_json['codebook_size'],
                beta=eval(codebook_training_data_json['beta']),
                prune_epochs=codebook_training_data_json['prune_epochs'],
                prune_values=[codebook_training_data_json['prune_value'] / len(codebook_training_data_json['prune_epochs'])] * len(codebook_training_data_json['prune_epochs']),
                prune_values_type=codebook_training_data_json['prune_value_type']
            )
        )

    if dataset == 'cifar10':
        resnet_with_codebook = create_resnet_with_codebook(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path), PSNR, codebook_training_data, train_dl)
    elif dataset == 'cifar100':
        resnet_with_codebook = create_resnet_with_codebook(cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path, num_classes=100), PSNR, codebook_training_data, train_dl)
    else:
        resnet_with_codebook = create_resnet_with_codebook(imagenet_resnet_loader_generator(model_name), PSNR, codebook_training_data, train_dl)

    if not pretrained_weights_path is None:
        resnet_with_codebook.load_state_dict(torch.load(pretrained_weights_path))

    all_params = list(resnet_with_codebook.parameters())

    codebook_params, codebook_ids = get_codebook_params_and_ids(resnet_with_codebook)
    non_codebook_params = [p for p in all_params if id(p) not in codebook_ids]

    optimizer = torch.optim.Adam([
        {'params': non_codebook_params, 'lr': non_codebook_lr},
        {'params': codebook_params, 'lr': codebook_lr},
    ])
    
    train_metrics, train_losses, valid_metrics, valid_losses, best_acc = train_model(n_epochs,
                                                                            resnet_with_codebook,
                                                                            resnet_vib_loss,
                                                                            train_dl,
                                                                            valid_dl,
                                                                            optimizer,
                                                                            codebook_training_data)

    tail_modules = [tail_module_name for tail_module_name, _ in resnet_with_codebook.named_modules()\
            if 'layer2' in tail_module_name or\
            'layer3' in tail_module_name or\
            'linear' in tail_module_name
            ]
    def return_example_input(input_res=(3, 32, 32)):
        return next(iter(train_dl))[0].to('cuda')
    def count_ops_and_params(model, ignore_list=[]):
        # return op_counter.count_ops_and_params(model, example_inputs=return_example_input(), ignore_list=ignore_list)
        return ptflops.get_model_complexity_info(model, input_res=tuple(next(iter(train_dl))[0].shape)[1:], print_per_layer_stat=False, as_strings=False, input_constructor=return_example_input, verbose=False, ignore_list=ignore_list)

    def recursive_dict():
        return defaultdict(recursive_dict)
    measures = recursive_dict()

    measures['codebook_model']['acc']['ori'], measures['codebook_model']['acc']['codebook'], measures['codebook_model']['acc']['best'] = evaluate_codebook_model(resnet_with_codebook, valid_dl, -1) * 100.0, evaluate_codebook_model(resnet_with_codebook, valid_dl, 0) * 100.0, best_acc * 100.0
    measures['codebook_model']['total']['flops'], measures['codebook_model']['total']['params'] = count_ops_and_params(resnet_with_codebook)
    measures['codebook_model']['head']['flops'], measures['codebook_model']['head']['params'] = count_ops_and_params(resnet_with_codebook, ignore_list=tail_modules)

    if dataset == "cifar10":
        unmodified_model = cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path)()
    elif dataset == "cifar100":
        unmodified_model = cifar_resnet_loader_generator(model_name, reference_pretrained_weights_path, num_classes=100)()
    measures['unmodified_model']['acc'] = evaluate_codebook_model(unmodified_model, valid_dl, -1) * 100.0
    measures['unmodified_model']['total']['flops'], measures['unmodified_model']['total']['params'] = count_ops_and_params(unmodified_model)
    measures['unmodified_model']['head']['flops'], measures['unmodified_model']['head']['params'] = count_ops_and_params(unmodified_model, ignore_list=tail_modules)

    measures['comm']['ori'], measures['comm']['comp'] = codebook_training_data[0].codebook_size, resnet_with_codebook(return_example_input()).codebook_outputs[0][2].n_embeddings
    measures['comm']['efficiency'] = (measures['comm']['ori'] / measures['comm']['comp']) * 100.0

    print(10 * '*' + 'Total Computation' + 10 * '*')
    print('Model with Codebook:')
    print(f"FLOPs: {measures['codebook_model']['total']['flops']}, Params: {measures['codebook_model']['total']['params']}")
    print('Unmodified Model:')
    print(f"FLOPs: {measures['unmodified_model']['total']['flops']}, Params: {measures['unmodified_model']['total']['params']}")

    print(10 * '*' + 'Head Computation' + 10 * '*')
    print('Model with Codebook:')
    print(f"FLOPs: {measures['codebook_model']['head']['flops']}, Params: {measures['codebook_model']['head']['params']}")
    print('Unmodified Model:')
    print(f"FLOPs: {measures['unmodified_model']['head']['flops']}, Params: {measures['unmodified_model']['head']['params']}")

    print(10 * '*' + 'Performance' + 10 * '*')
    print(f"Original: {measures['codebook_model']['acc']['ori']} %")
    print(f"Codebook: {measures['codebook_model']['acc']['codebook']} %")
    print(f"Unmodified: {measures['unmodified_model']['acc']} %")

    print(10 * '*' + 'Three-way Trade-off' + 10 * '*')
    print('Communication:')
    print(f"Efficiency: {measures['comm']['ori']} / {measures['comm']['comp']} = {measures['comm']['efficiency']} %")
    print('Performance:')
    print(f"Accuracy: {measures['codebook_model']['acc']['best']} - {measures['unmodified_model']['acc']} = {measures['codebook_model']['acc']['best'] - measures['unmodified_model']['acc']} %")

    training_data = {
        'train_metrics': train_metrics,
        'train_losses': train_losses,
        'valid_metrics': valid_metrics,
        'valid_losses': valid_losses,
        'best_acc': best_acc
    }

    with open(output_dir + '/training_data.json', 'w') as outfile:
        json.dump(training_data, outfile)

    with open(output_dir + '/measures.json', 'w') as outfile:
        json.dump(measures, outfile, indent=4)

    if save_weights_after_train:
        torch.save(resnet_with_codebook.state_dict(), f'{output_dir}/model.pth')
    
