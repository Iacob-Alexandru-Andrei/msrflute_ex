import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

import importlib.util
from experiments.cv_resnet_fedcifar100.group_normalization import GroupNorm2d
from core.model import BaseModel

import sys

sys.path.append("/home/aai30/nfs-share/projects")


from FedScale.fedscale.utils.models.specialized.resnet_speech import resnet34


""" 
    The ResNet models are taken from FedML repository. For more information regarding this model, 
    please refer to https://github.com/FedML-AI/FedML/blob/master/python/fedml/model/cv/resnet_gn.py.
"""


class RESNET(BaseModel):
    """This is a PyTorch model with some extra methods"""

    def __init__(self, model_config):
        super().__init__()
        self.net = resnet34(num_classes=35, in_channels=1)

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        """Performs forward step and computes the loss"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features, labels = input["x"].to(device), input["y"].to(device)
        output = self.net.forward(features)

        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        """Performs forward step and computes metrics"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        features, labels = input["x"].to(device), input["y"].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()

        return {"output": output, "acc": accuracy, "batch_size": n_samples}
