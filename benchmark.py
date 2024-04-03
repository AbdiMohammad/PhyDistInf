import json
import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

from PhyDistInf import CodebookTrainData, cifar_resnet_loader_generator, imagenet_resnet_loader_generator, create_resnet_with_codebook, get_codebook_params_and_ids, evaluate_codebook_model
from PhyDistInf import ConstantCoefficient, LinearCoefficient
from codebook_output import CodebookOutput

def measure_inference_time(model, dataloader):
    device = next(model.parameters()).device

    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    head_inference_time, tail_inference_time = [], []

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
                if type(out) == CodebookOutput:
                    transmit_event = out.codebook_outputs[0][2].transmit_event
                    receive_event = out.codebook_outputs[0][2].receive_event
                head_inference_time.append(start_event.elapsed_time(transmit_event))
                tail_inference_time.append(receive_event.elapsed_time(end_event))

    head_inference_time = torch.Tensor(head_inference_time)
    tail_inference_time = torch.Tensor(tail_inference_time)
    print(f"head inference time averaged over {head_inference_time.shape}: {head_inference_time.mean()}")
    print(f"tail inference time averaged over {tail_inference_time.shape}: {tail_inference_time.mean()}")

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

if __name__ == '__main__':

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    dataset = json_data["dataset"]
    BATCH_SIZE = 1024
    model_name = json_data["model"]

    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10': 
        cifar10_test = torchvision.datasets.CIFAR10('dataset/cifar10', train=False, download=True, transform=transform)
        cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=BATCH_SIZE, num_workers=16)

        valid_dl = cifar10_test_dataloader

    if dataset == 'imagenet':
        # TODO: load actual imagenet
        imagenet_test = TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
        imagenet_test_dataloader = DataLoader(imagenet_test, batch_size=BATCH_SIZE, num_workers=16)

        valid_dl = imagenet_test_dataloader

    PSNR = json_data['PSNR']
    pretrained_weights_path = f"{json_data['output_folder']}/model.pth"
    pretrained_weights = torch.load(pretrained_weights_path, map_location=torch.device('cuda'))
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
        resnet_with_codebook = create_resnet_with_codebook(cifar_resnet_loader_generator(model_name), PSNR, codebook_training_data)
    else:
        resnet_with_codebook = create_resnet_with_codebook(imagenet_resnet_loader_generator(model_name), PSNR, codebook_training_data)

    resnet_with_codebook.load_state_dict(pretrained_weights)

    all_params = list(resnet_with_codebook.parameters())

    codebook_params, codebook_ids = get_codebook_params_and_ids(resnet_with_codebook)
    non_codebook_params = [p for p in all_params if id(p) not in codebook_ids]

    optimizer = torch.optim.Adam([
        {'params': non_codebook_params, 'lr': non_codebook_lr},
        {'params': codebook_params, 'lr': codebook_lr},
    ])




    measure_inference_time(resnet_with_codebook, valid_dl)
    profile_model(resnet_with_codebook, valid_dl)