#%%
import torch
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.models import resnet18, resnet152
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torch.utils.data import DataLoader
from codebook_output import CodebookOutput
# %%

class ImageNetModifiedBasicBlock(BasicBlock):

    def forward(self, x):
        if type(x) == CodebookOutput:
            return x.map(super().forward)
        else:
            return super().forward(x)

class ImageNetModifiedBottleneck(Bottleneck):

    def forward(self, x):
        if type(x) == CodebookOutput:
            return x.map(super().forward)
        else:
            return super().forward(x)
        
class ImageNetModifiedResnet(ResNet):

    def backend(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if type(x) == CodebookOutput:
            x = x.map(self.backend)
        else:
            x = self.backend(x)

        return x

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ImageNetModifiedResnet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def imagenet_resnet18(*, progress: bool = True, **kwargs: Any) -> ImageNetModifiedResnet:
    weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
    return _resnet(ImageNetModifiedBasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def imagenet_resnet34(*, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet34_Weights.verify(ResNet34_Weights.IMAGENET1K_V1)

    return _resnet(ImageNetModifiedBasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def imagenet_resnet50(*, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V1)
    return _resnet(ImageNetModifiedBottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def imagenet_resnet101(*, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet101_Weights.verify(ResNet101_Weights.IMAGENET1K_V1)
    return _resnet(ImageNetModifiedBottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def imagenet_resnet152(*, progress: bool = True, **kwargs: Any) -> ResNet:
    weights = ResNet152_Weights.verify(ResNet152_Weights.IMAGENET1K_V1)
    return _resnet(ImageNetModifiedBottleneck, [3, 8, 36, 3], weights, progress, **kwargs)
