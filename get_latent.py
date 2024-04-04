import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image

from PhyDistInf import cifar_resnet_loader_generator, StopCompute, model_device
from codebook_output import CodebookOutput

def get_layer_output(input, model, layer_name):
    exec('model.' + layer_name + '= StopCompute(model.' + layer_name + ')')

    try:
        _ = model(input.to(model_device(model)))
    except Exception as e:
        if type(e.args[0]) == CodebookOutput:
            return e.args[0].original_tensor.detach()
        else:
            return e.args[0].detach()
    finally:
        exec('model.' + layer_name + '= model.' + layer_name + '.inner')

if __name__ == "__main__":

    model_name = sys.argv[1]
    image_filedir = sys.argv[2]
    latent_layer_name = sys.argv[3]

    normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
    ])

    images = {}
    image_filenames = os.listdir(image_filedir)
    for image_filename in image_filenames:
        with Image.open(os.path.join(image_filedir, image_filename)) as image:
            images[f"{image_filename.split('.')[0]}"] = transform(image).unsqueeze(0)

    device = torch.device('cuda')

    model = cifar_resnet_loader_generator(model_name, f"resource/cifar_pretrained/cifar10/{model_name}.pth")().to(device)

    output_dir = os.path.join(image_filedir, "latent_SC")
    os.mkdir(output_dir)

    latent_list = {}
    for name, image in images.items():
        image = image.to(device)
        latent_list[name] = get_layer_output(image, model, latent_layer_name)
        torch.save(latent_list[name], os.path.join(output_dir, f"{name}.pth"))
        np.save(os.path.join(output_dir, name), latent_list[name].cpu().numpy())
        